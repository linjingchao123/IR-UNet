import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
config = {}
config['anchors'] = [5., 10., 20.] #[ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 2
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
debug = True #True #True#False #True
class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        self.output = nn.Sequential(nn.Conv3d(32, 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))
        self.att1_1 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att1_2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att1_3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att2_1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att2_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att3_1 = Attention_block(F_g=128, F_l=128, F_int=64)

        self.up1 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv1 = conv_block(ch_in=64, ch_out=32)
        self.up2= up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self,x,coord):

        x1_0 = F.relu(F.max_pool3d(self.encoder1(x),2,2))

        x2_0 = F.relu(F.max_pool3d(self.encoder2(x1_0),2,2))

        at1_1 = self.att1_1(g=self.up1(x2_0), x=x1_0)
        x1_1 = self.Up_conv1(torch.cat([at1_1, self.up1(x2_0)], 1))

        x3_0 = F.relu(F.max_pool3d(self.encoder3(x2_0),2,2))

        at2_1 = self.att2_1(g=self.up2(x3_0), x=x2_0)
        x2_1 = self.Up_conv2(torch.cat([at2_1, self.up2(x3_0)], 1))

        at1_2 = self.att1_2(g=self.up1(x2_1), x=x1_1)
        x1_2 = self.Up_conv1(torch.cat([at1_2, self.up1(x2_1)], 1))

        x4_0 = F.relu(F.max_pool3d(self.encoder4(x3_0),2,2))

        at3_1 = self.att3_1(g=self.up3(x4_0), x=x3_0)
        x3_1 = self.Up_conv3(torch.cat([at3_1, self.up3(x4_0)], 1))

        at2_2 = self.att2_2(g=self.up2(x3_1), x=x2_1)
        x2_2 = self.Up_conv2(torch.cat([at2_2, self.up2(x3_1)], 1))

        at1_3 = self.att1_3(g=self.up1(x2_2), x=x1_2)
        x1_3 = self.Up_conv1(torch.cat([at1_3, self.up1(x2_2)], 1))





        # up4_0 = F.relu(F.interpolate(self.decoder2(x4_0),scale_factor=(2,2,2),mode ='trilinear'))
        # x3_1 = torch.add(up4_0,x3_0)
        # print('x3_1:', x3_1.shape)
        # output2 = self.map2(out)
        # print('output2:', output2.shape)
        # up3_1 = F.relu(F.interpolate(self.decoder3(x3_1),scale_factor=(2,2,2),mode ='trilinear'))
        # x2_2 = torch.add(up3_1,x2_0)
        # print('x2_2:', x2_2.shape)
        # # output3 = self.map3(out)
        # # print('output3:', output3.shape)
        # up2_2 = F.relu(F.interpolate(self.decoder4(x2_2),scale_factor=(2,2,2),mode ='trilinear'))
        # x1_3 = torch.add(up2_2,x1_0)
        # print('x1_3:', x1_3.shape)
        # out4 = self.decoder5(out3)
        # out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        # output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        #     return output4
        out=self.output(x1_3)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        print('11', out.size())
        # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(1, 48, 48, 48, len(config['anchors']), 5)
        print('12', out.size())
        return out

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


def get_model():
    net = UNet()
    print('Model----UNETplus!')
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb


def test():
    debug = True
    net = UNet()
    x = Variable(torch.randn(1,1,96,96,96))
    crd = Variable(torch.randn(1,3,48,48,48))
    y = net(x, crd)
    #print(y)
test()