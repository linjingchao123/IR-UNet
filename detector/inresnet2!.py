import torch
from torch import nn
from layers import *
from torch.autograd import Variable

config = {}
config['anchors'] = [5., 10., 20.] #[ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5 #3 #6. #mm
config['sizelim2'] = 10 #30
config['sizelim3'] = 20 #40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}


config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']

#config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','d92998a73d4654a442e6d6ba15bbb827','990fbe3f0a1b53878669967b9afd1441','820245d8b211808bd18e78ff5be16fdb','adc3bbc63d40f8761c59be10f1e504c3',
#                       '417','077','188','876','057','087','130','468']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        self.forw1 = inRes(n_in=24,n_out=16)
        self.forw2 = inRes(n_in=32, n_out=32)
        self.forw3 = inRes(n_in=64, n_out=32)
        self.forw4 = inRes(n_in=64, n_out=32)
        self.forw5 = inRes(n_in=32, n_out=16)

        self.back1 = inRes(n_in=128, n_out=32)
        self.back2 = inRes(n_in=131, n_out=64)
        self.back3 = inRes(n_in=64, n_out=32)
        self.back4 = inRes(n_in=128, n_out=64)
        self.back5 = inRes(n_in=131, n_out=32)


        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,16,32,32,32]
        self.featureNum_back =    [64,32,32]
        # for i in range(len(num_blocks_forw)):
        #     blocks = []
        #     for j in range(num_blocks_forw[i]):
        #         if j == 0:
        #             blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
        #         else:
        #             blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
        #     setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))
        #
        # for i in range(len(num_blocks_back)):
        #     blocks = []
        #     for j in range(num_blocks_back[i]):
        #         if j == 0:
        #             if i==0:
        #                 addition = 3
        #             else:
        #                 addition = 0
        #             blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
        #         else:
        #             blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
        #     setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(128, 64, kernel_size = 1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x, coord):
        # print('x, coord',x.shape, coord.shape)#torch.Size([1, 1, 96, 96, 96]) torch.Size([BS, 3, 24, 24, 24])
        out = self.preBlock(x)#torch.Size([1, 24, 96, 96, 96])
        
        out_pool,indices0 = self.maxpool1(out)#torch.Size([1, 24, 48, 48, 48])
        out1 = self.forw1(out_pool)  #out1.size(1,32,48,48,48)
        out1 = self.forw5(out1)#out1.Size([1, 32, 48, 48, 48])
        out1_pool,indices1 = self.maxpool2(out1)#torch.Size([1, 32, 24, 24, 24])

        out2 = self.forw2(out1_pool) #out2.Size([1, 64, 24, 24, 24])
        out2 = self.forw3(out2) #out2.Size([1, 64, 24, 24, 24])#out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)#torch.Size([1, 64, 12, 12, 12])

        out3 = self.forw3(out2_pool) #out3.Size([1, 64, 12, 12, 12])
        out3 = self.forw3(out3) #out3.Size([1, 64, 12, 12, 12])
        out3 = self.forw3(out3) #out3.Size([1, 64, 12, 12, 12])
        rev1 = self.path3(out3)
        comb1 = self.back5(torch.cat((rev1, out2,coord), 1))
        comb1 = self.back3(comb1)
        comb1 = self.back3(comb1)

        out3_pool,indices3 = self.maxpool4(out3)#torch.Size([1, 64, 6, 6, 6])
        out4 = self.forw4(out3_pool) #out4.Size([1, 64, 6, 6, 6])#out4 = self.drop(out4)
        out4 = self.forw4(out4) #out4.Size([1, 64, 6, 6, 6])
        out4 = self.forw4(out4) #out4.Size([1, 64, 6, 6, 6])

        # print('out',out_pool.shape,out1.shape,out1_pool.shape,out2.shape,out2_pool.shape,out3.shape,out3_pool.shape,out4.shape)
        rev3 = self.path1(out4)#torch.Size([1, 64, 12, 12, 12])
        comb3 = self.back1(torch.cat((rev3, out3), 1))#comb3.Size([1, 64, 12, 12, 12])#comb3 = self.drop(comb3)
        comb3 = self.back3(comb3)#comb3.Size([1, 64, 12, 12, 12])
        comb3 = self.back3(comb3)#comb3.Size([1, 64, 12, 12, 12])

        rev2 = self.path2(comb3)#torch.Size([1, 64, 24, 24, 24])
        comb2 = self.back2(torch.cat((rev2, comb1,coord), 1))#comb2.Size([1, 128, 24, 24, 24])#64+64
        comb2 = self.back4(comb2)#comb2.Size([1, 128, 24, 24, 24])
        comb2 = self.back4(comb2)#comb2.Size([1, 128, 24, 24, 24])

        comb2 = self.drop(comb2)#comb2.Size([1, 128, 24, 24, 24])
        out = self.output(comb2)#out.Size([1, 15, 24, 24, 24])
        size = out.size()
        # print('out',out.shape)
        out = out.view(out.size(0), out.size(1), -1)#torch.Size([1, 15, 13824])
        print('out',out.shape)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        
        return out #out.Size([1, 24, 24, 24, 3, 5])

class inRes(nn.Module):
    expand = 2
    def __init__(self, n_in, n_out, stride=1):
        super(inRes, self).__init__()

        self.convl1 = nn.Conv3d(n_in, n_out, kernel_size=1, stride=1, padding=1)
        self.bnl1 = nn.BatchNorm3d(n_out)
        self.convl2 = nn.Conv3d(n_out, n_out, kernel_size=5,stride=1, padding=1)
        self.bnl2 = nn.BatchNorm3d(n_out)

        self.convr1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bnr1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.convr2 = nn.Conv3d(n_out, n_out, kernel_size=3, stride=1,padding=1)
        self.bnr2 = nn.BatchNorm3d(n_out)

        self.se = SELayer(n_out)

        if stride != 1 or n_out*self.expand != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out*self.expand, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out*self.expand))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        outl = self.convl1(x)
        outl = self.bnl1(outl)
        outl = self.relu(outl)
        outl = self.convl2(outl)
        outl = self.bnl2(outl)
        outl = self.se(outl)


        outr = self.convr1(x)
        outr = self.bnr1(outr)
        outr = self.relu(outr)
        outr = self.convr2(outr)
        outr = self.bnr2(outr)
        outr = self.se(outr)

        out = torch.cat((outl,outr),1)
        out += residual
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

def get_model():
    net = Net()
    # print('Net----res18!', net)
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb

def test():
    debug = True
    net = Net()
    x = Variable(torch.randn(1,1,96,96,96))
    crd = Variable(torch.randn(1,3,24,24,24))
    y = net(x, crd)
    # print(y)
test()