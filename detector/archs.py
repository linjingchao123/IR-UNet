import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
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
debug = True #True #True#False #True


# __all__ = ['UNet', 'NestedUNet']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super(NestedUNet,self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up1 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv1 = conv_block(ch_in=64, ch_out=32)

        self.up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att0_1 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att0_2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att0_3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att0_4 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.att1_1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att1_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att1_3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att2_1 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.att2_2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.att3_1 = Attention_block(F_g=256, F_l=256, F_int=128)


        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input, coord):
        print('x, coord', input.shape, coord.shape)
        print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        print('x0_0:', x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        print('x1_0:', x1_0.shape)
        at0_1 = self.att0_1(g=self.up1(x1_0), x=x0_0)
        x0_1 = self.Up_conv1(torch.cat([at0_1, self.up1(x1_0)], 1))
        print('x0_1:', x0_1.shape)
        x2_0 = self.conv2_0(self.pool(x1_0))
        at1_1 = self.att1_1(g=self.up2(x2_0),x=x1_0)
        x1_1 = self.Up_conv2(torch.cat([at1_1, self.up2(x2_0)], 1))
        at0_2 = self.att0_2(g=self.up1(x1_1), x=x0_1)
        x0_2 = self.Up_conv1(torch.cat([at0_2, self.up1(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        at2_1 = self.att2_1(g=self.up3(x3_0), x=x2_0)
        x2_1 = self.Up_conv3(torch.cat([at2_1, self.up3(x3_0)], 1))
        at1_2 = self.att1_2(g=self.up2(x2_1), x=x1_1)
        x1_2 = self.Up_conv2(torch.cat([at1_2, self.up2(x2_1)], 1))
        at0_3 = self.att0_3(g=self.up1(x1_2), x=x0_2)
        x0_3 = self.Up_conv1(torch.cat([at0_3, self.up1(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        at3_1 = self.att3_1(g=self.up4(x4_0), x=x3_0)
        x3_1 = self.Up_conv4(torch.cat([at3_1, self.up4(x4_0)], 1))
        at2_2 = self.att2_2(g=self.up3(x3_1), x=x2_1)
        x2_2 = self.Up_conv3(torch.cat([at2_2, self.up3(x3_1)], 1))
        at1_3 = self.att1_3(g=self.up2(x2_2), x=x1_2)
        x1_3 = self.Up_conv2(torch.cat([at1_3, self.up2(x2_2)], 1))
        at0_4 = self.att0_4(g=self.up1(x1_3), x=x0_3)
        x0_4 = self.Up_conv1(torch.cat([at0_4, self.up1(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            out = self.final(x0_4)
            return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
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
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


def get_model():
    net = NestedUNet()
    print('Model----UNETplus!')
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb


# def test():
#     debug = True
#     net = UNETplus()
#     x = Variable(torch.randn(1,1,96,96,96))
#     crd = Variable(torch.randn(1,3,24,24,24))
#     y = net(x, crd)
#     # print(y)
#
# test()