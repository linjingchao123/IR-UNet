
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']
        self.layer1 = crossnet1(n_in=64,n_out=324)
        self.layer2 = crossnet2(n_in=324, n_out=672)
        self.layer3 = crossnet3(n_in=672, n_out=1530)
        self.layer4 = crossnet4(n_in=1530, n_out=2559)

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        # self.last_planes = 64
        # self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        # self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        # self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        # self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(2559, 2)#10)


    def forward(self, x):

        out = self.conv1(x)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool3d(out, 4)

        out_1 = out.view(out.size(0), -1)

        out = self.linear(out_1)

        return out, out_1

class crossnet1(nn.Module):
    expand = 2
    def __init__(self, n_in, n_out, stride=1):
        super(crossnet1, self).__init__()

        self.convup1= nn.Conv3d(64, 96, kernel_size=1, stride=1, padding=0)
        self.bnup1 = nn.BatchNorm3d(96)
        self.convup2 = nn.Conv3d(96, 96, kernel_size=3,stride=1, padding=1)
        self.bnup2 = nn.BatchNorm3d(96)
        self.convup3 = nn.Conv3d(96, 108, kernel_size=1, stride=1, padding=0)
        self.bnup3 = nn.BatchNorm3d(108)

        self.convmid1 = nn.Conv3d(64, 96, kernel_size=1, stride=1, padding=0)
        self.bnmid1 = nn.BatchNorm3d(96)
        self.relu = nn.ReLU(inplace=True)
        self.convmid2 = nn.Conv3d(96, 96, kernel_size=5, stride=1,padding=1)
        self.bnmid2 = nn.BatchNorm3d(96)
        self.convmid3 = nn.Conv3d(96, 108, kernel_size=1, stride=1, padding=1)
        self.bnmid3 = nn.BatchNorm3d(108)

        self.convdown1 = nn.Conv3d(64, 96, kernel_size=1, stride=1, padding=1)
        self.bndown1 = nn.BatchNorm3d(96)
        self.convdown2 = nn.Conv3d(96, 96, kernel_size=7, stride=1, padding=1)
        self.bndown2 = nn.BatchNorm3d(96)
        self.convdown3 = nn.Conv3d(96, 108, kernel_size=1, stride=1, padding=1)
        self.bndown3 = nn.BatchNorm3d(108)

        self.se = SELayer(96)

        self.shortcut = nn.Sequential(
            nn.Conv3d(64, 108, kernel_size=1, stride=stride,padding=0),
            nn.BatchNorm3d(108))


    def forward(self, x):
        residual = self.shortcut(x)

        outup = self.convup1(x)
        outup = self.bnup1(outup)
        outup = self.relu(outup)

        outup = self.convup2(outup)
        outup = self.bnup2(outup)
        outup = self.relu(outup)
        outup = self.se(outup)

        outup = self.convup3(outup)
        outup = self.bnup3(outup)


        outmid = self.convmid1(x)
        outmid = self.bnmid1(outmid)
        outmid = self.relu(outmid)

        outmid = self.convmid2(outmid)
        outmid = self.bnmid2(outmid)
        outmid = self.relu(outmid)
        outmid = self.se(outmid)

        outmid = self.convmid3(outmid)
        outmid = self.bnmid3(outmid)


        outdown = self.convdown1(x)
        outdown = self.bndown1(outdown)
        outdown = self.relu(outdown)

        outdown = self.convdown2(outdown)
        outdown = self.bndown2(outdown)
        outdown = self.relu(outdown)
        outdown = self.se(outdown)

        outdown = self.convdown3(outdown)
        outdown = self.bndown3(outdown)


        out1 = outup+residual+outdown
        out2 = outmid+residual++outup
        out3 = outdown+residual+outmid
        out = torch.cat((out1,out2,out3),1)
        out = self.relu(out)

        return out

class crossnet2(nn.Module):
    expand = 2
    def __init__(self, n_in, n_out, stride=1):
        super(crossnet2, self).__init__()

        self.convup1= nn.Conv3d(324, 192, kernel_size=1, stride=1, padding=0)
        self.bnup1 = nn.BatchNorm3d(192)
        self.convup2 = nn.Conv3d(192, 192, kernel_size=3,stride=2 ,padding=1)
        self.bnup2 = nn.BatchNorm3d(192)
        self.convup3 = nn.Conv3d(192, 224, kernel_size=1, stride=1, padding=0)
        self.bnup3 = nn.BatchNorm3d(224)

        self.convmid1 = nn.Conv3d(324, 192, kernel_size=1, stride=1,padding=1)
        self.bnmid1 = nn.BatchNorm3d(192)
        self.relu = nn.ReLU(inplace=True)
        self.convmid2 = nn.Conv3d(192, 192, kernel_size=5, stride=2,padding=1)
        self.bnmid2 = nn.BatchNorm3d(192)
        self.convmid3 = nn.Conv3d(192, 224, kernel_size=1, stride=1, padding=0)
        self.bnmid3 = nn.BatchNorm3d(224)

        self.convdown1 = nn.Conv3d(324, 192, kernel_size=1, stride=1, padding=1)
        self.bndown1 = nn.BatchNorm3d(192)
        self.convdown2 = nn.Conv3d(192, 192, kernel_size=7, stride=2, padding=2)
        self.bndown2 = nn.BatchNorm3d(192)
        self.convdown3 = nn.Conv3d(192, 224, kernel_size=1, stride=1, padding=0)
        self.bndown3 = nn.BatchNorm3d(224)

        self.se = SELayer(192)

        self.shortcut = nn.Sequential(
            nn.Conv3d(324, 224, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm3d(224))

    def forward(self, x):
        residual = self.shortcut(x)

        outup = self.convup1(x)
        outup = self.bnup1(outup)
        outup = self.relu(outup)

        outup = self.convup2(outup)
        outup = self.bnup2(outup)
        outup = self.relu(outup)
        outup = self.se(outup)

        outup = self.convup3(outup)
        outup = self.bnup3(outup)


        outmid = self.convmid1(x)
        outmid = self.bnmid1(outmid)
        outmid = self.relu(outmid)

        outmid = self.convmid2(outmid)
        outmid = self.bnmid2(outmid)
        outmid = self.relu(outmid)
        outmid = self.se(outmid)

        outmid = self.convmid3(outmid)
        outmid = self.bnmid3(outmid)


        outdown = self.convdown1(x)
        outdown = self.bndown1(outdown)
        outdown = self.relu(outdown)

        outdown = self.convdown2(outdown)
        outdown = self.bndown2(outdown)
        outdown = self.relu(outdown)
        outdown = self.se(outdown)

        outdown = self.convdown3(outdown)
        outdown = self.bndown3(outdown)


        out1 = outup+residual+outdown
        out2 = outmid+residual++outup
        out3 = outdown+residual+outmid
        out = torch.cat((out1,out2,out3),1)
        out = self.relu(out)

        return out

class crossnet3(nn.Module):
    expand = 2
    def __init__(self, n_in, n_out, stride=1):
        super(crossnet3, self).__init__()

        self.convup1= nn.Conv3d(672, 384, kernel_size=1, stride=1, padding=0)
        self.bnup1 = nn.BatchNorm3d(384)
        self.convup2 = nn.Conv3d(384, 384, kernel_size=3,stride=2 ,padding=1)
        self.bnup2 = nn.BatchNorm3d(384)
        self.convup3 = nn.Conv3d(384, 510, kernel_size=1, stride=1, padding=0)
        self.bnup3 = nn.BatchNorm3d(510)

        self.convmid1 = nn.Conv3d(672, 384, kernel_size=1, stride=1,padding=1)
        self.bnmid1 = nn.BatchNorm3d(384)
        self.relu = nn.ReLU(inplace=True)
        self.convmid2 = nn.Conv3d(384, 384, kernel_size=5, stride=2,padding=1)
        self.bnmid2 = nn.BatchNorm3d(384)
        self.convmid3 = nn.Conv3d(384, 510, kernel_size=1, stride=1, padding=0)
        self.bnmid3 = nn.BatchNorm3d(510)

        self.convdown1 = nn.Conv3d(672, 384, kernel_size=1, stride=1, padding=1)
        self.bndown1 = nn.BatchNorm3d(384)
        self.convdown2 = nn.Conv3d(384, 384, kernel_size=7, stride=2, padding=2)
        self.bndown2 = nn.BatchNorm3d(384)
        self.convdown3 = nn.Conv3d(384, 510, kernel_size=1, stride=1, padding=0)
        self.bndown3 = nn.BatchNorm3d(510)

        self.se = SELayer(384)

        self.shortcut = nn.Sequential(
            nn.Conv3d(672, 510, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm3d(510))

    def forward(self, x):
        residual = self.shortcut(x)

        outup = self.convup1(x)
        outup = self.bnup1(outup)
        outup = self.relu(outup)

        outup = self.convup2(outup)
        outup = self.bnup2(outup)
        outup = self.relu(outup)
        outup = self.se(outup)

        outup = self.convup3(outup)
        outup = self.bnup3(outup)


        outmid = self.convmid1(x)
        outmid = self.bnmid1(outmid)
        outmid = self.relu(outmid)

        outmid = self.convmid2(outmid)
        outmid = self.bnmid2(outmid)
        outmid = self.relu(outmid)
        outmid = self.se(outmid)

        outmid = self.convmid3(outmid)
        outmid = self.bnmid3(outmid)


        outdown = self.convdown1(x)
        outdown = self.bndown1(outdown)
        outdown = self.relu(outdown)

        outdown = self.convdown2(outdown)
        outdown = self.bndown2(outdown)
        outdown = self.relu(outdown)
        outdown = self.se(outdown)

        outdown = self.convdown3(outdown)
        outdown = self.bndown3(outdown)


        out1 = outup+residual+outdown
        out2 = outmid+residual++outup
        out3 = outdown+residual+outmid
        out = torch.cat((out1,out2,out3),1)
        out = self.relu(out)

        return out

class crossnet4(nn.Module):
    expand = 2
    def __init__(self, n_in, n_out, stride=1):
        super(crossnet4, self).__init__()

        self.convup1= nn.Conv3d(1530, 768, kernel_size=1, stride=1, padding=0)
        self.bnup1 = nn.BatchNorm3d(768)
        self.convup2 = nn.Conv3d(768, 768, kernel_size=3,stride=2 ,padding=1)
        self.bnup2 = nn.BatchNorm3d(768)
        self.convup3 = nn.Conv3d(768, 853, kernel_size=1, stride=1, padding=0)
        self.bnup3 = nn.BatchNorm3d(853)

        self.convmid1 = nn.Conv3d(1530, 768, kernel_size=1, stride=1,padding=1)
        self.bnmid1 = nn.BatchNorm3d(768)
        self.relu = nn.ReLU(inplace=True)
        self.convmid2 = nn.Conv3d(768, 768, kernel_size=5, stride=2,padding=1)
        self.bnmid2 = nn.BatchNorm3d(768)
        self.convmid3 = nn.Conv3d(768, 853, kernel_size=1, stride=1, padding=0)
        self.bnmid3 = nn.BatchNorm3d(853)

        self.convdown1 = nn.Conv3d(1530, 768, kernel_size=1, stride=1, padding=1)
        self.bndown1 = nn.BatchNorm3d(768)
        self.convdown2 = nn.Conv3d(768, 768, kernel_size=7, stride=2, padding=2)
        self.bndown2 = nn.BatchNorm3d(768)
        self.convdown3 = nn.Conv3d(768, 853, kernel_size=1, stride=1, padding=0)
        self.bndown3 = nn.BatchNorm3d(853)

        self.se = SELayer(768)

        self.shortcut = nn.Sequential(
            nn.Conv3d(1530, 853, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm3d(853))

    def forward(self, x):
        residual = self.shortcut(x)

        outup = self.convup1(x)
        outup = self.bnup1(outup)
        outup = self.relu(outup)

        outup = self.convup2(outup)
        outup = self.bnup2(outup)
        outup = self.relu(outup)
        outup = self.se(outup)

        outup = self.convup3(outup)
        outup = self.bnup3(outup)


        outmid = self.convmid1(x)
        outmid = self.bnmid1(outmid)
        outmid = self.relu(outmid)

        outmid = self.convmid2(outmid)
        outmid = self.bnmid2(outmid)
        outmid = self.relu(outmid)
        outmid = self.se(outmid)

        outmid = self.convmid3(outmid)
        outmid = self.bnmid3(outmid)


        outdown = self.convdown1(x)
        outdown = self.bndown1(outdown)
        outdown = self.relu(outdown)

        outdown = self.convdown2(outdown)
        outdown = self.bndown2(outdown)
        outdown = self.relu(outdown)
        outdown = self.se(outdown)

        outdown = self.convdown3(outdown)
        outdown = self.bndown3(outdown)


        out1 = outup+residual+outdown
        out2 = outmid+residual++outup
        out3 = outdown+residual+outmid
        out = torch.cat((out1,out2,out3),1)
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

def DPN92_3D():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)