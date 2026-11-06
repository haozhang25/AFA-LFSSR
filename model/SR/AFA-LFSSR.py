import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        n_group = 5
        n_block = 5
        self.angRes = args.angRes_in
        self.factor = args.scale_factor

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)

        ########################### UP-Sampling #############################
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x, info=None):
        x = rearrange(x, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        [b, c, u, v, h, w] = x.size()

        x_upscale = LFS_interpolate(x, scale_factor=self.factor, mode='bicubic')
        x_upscale = rearrange(x_upscale, 'b c u v h w -> b c (u h) (v w)', u=u, v=v)

        # Initial Feature Extraction
        x = rearrange(x, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        # Deep Spatial-Angular Correlation Learning
        buffer = rearrange(buffer, "b c (u v) h w -> b c (h u) (w v)", u=self.angRes, v=self.angRes)
        buffer = self.disentg(buffer)

        # UP-Sampling
        buffer = rearrange(buffer, "b c (h u) (w v) -> b c (u h) (v w)", u=self.angRes, v=self.angRes)
        buffer = self.upsample(buffer)
        out = buffer + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()

        # intra spatial feature
        self.IntraSpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False)
        )

        # inter spatial feature
        self.InterSpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False)
        )

        # intra angular feature
        self.IntraAngConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
                      padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(angRes),
        )

        # inter angular feature
        self.InterAngConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes * 2), stride=int(angRes * 2),
                      dilation=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, int(4 * angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2 * angRes),
        )

        # horizontal EPI feature
        self.HEPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False)
        )

        # vertical EPI feature
        self.VEPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False)
        )

    def forward(self, buffer):
        feaIntraSpa = self.IntraSpaConv(buffer)
        buffer = feaIntraSpa + buffer
        feaInterSpa = self.InterSpaConv(buffer)
        buffer = feaInterSpa + buffer

        feaIntraAng = self.IntraAngConv(buffer)
        buffer = feaIntraAng + buffer
        feaInterAng = self.InterAngConv(buffer)
        buffer = feaInterAng + buffer

        feaHEPI = self.HEPIConv(buffer)
        buffer = feaHEPI + buffer
        feaVEPI = self.VEPIConv(buffer)
        out = feaVEPI + buffer
        return out

def LFS_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_upscale

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out, HR)

        return loss


def weights_init(m):
    pass


if __name__ == '__main__':
    from argparse import Namespace
    from thop import profile

    args = Namespace(angRes_in=5, scale_factor=2)
    net = get_model(args).cuda()
    input = torch.randn(1, 1, 5*32, 5*32).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))