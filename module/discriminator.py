#encoding:utf-8

from .importer import *
from .base_module import *

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=5):
        super(Discriminator,self).__init__()
        #レイヤーは全部でn_layers個
        #最初の1レイヤー目
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        #(n_layers-2)個の中間のレイヤー
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        #最後のレイヤー
        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
                  
        self.model = nn.Sequential(*model)

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, input):
        #論文中で、Encoderと呼ばれている層を通す
        x = self.model(input)

        #平均を取る操作によって
        # x  : torch.Size([batch_size,channel,Height,Width])を
        #gap : torch.Size([batch_size,channel,1,1])に変換する
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        #gap_logit : torch.Size([batch_size,1])
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        #gap_weight : torch.Size([1,channel])
        gap_weight = list(self.gap_fc.parameters())[0]
        #gap_weight.unsqueeze(2).unsqueeze(3) : torch.Size([1,channel,1,1])
        #gap : torch.Size([batch_size,channel,Height,Width])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        #最大値を取る操作によって
        # x  : torch.Size([batch_size,channel,Height,Width])を
        #gmp : torch.Size([batch_size,channel,1,1])に変換する
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        #gmp_logit : torch.Size([batch_size,1])
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        #gmp_weight : torch.Size([1,channel])
        gmp_weight = list(self.gmp_fc.parameters())[0] #元々[1]以降は存在しない　つまりlen(list(self.gmp_fc.parameters())) = 1
        #gmp_weight.unsqueeze(2).unsqueeze(3) : torch.Size([1,channel,1,1])
        #gmp : torch.Size([batch_size,channel,Height,Width])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        #cam_logit : torch.Size([batch_size,2])
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        #この時点で
        # x : torch.Size([batch_size,channel*2,Height,Width])
        x = self.leaky_relu(self.conv1x1(x))
        #この時点で
        # x : torch.Size([batch_size,channel,Height,Width])

        #heatmap : torch.Size([batch_size,1,Height,Width])
        heatmap = torch.sum(x, dim=1, keepdim=True)

        #self.pad = nn.ReflectionPad2d(1)を適用
        x = self.pad(x)
        #この時点で
        # x : torch.Size([batch_size,channel,Height+2,Width+2])
        out = self.conv(x)
        #out : torch.Size([batch_size,1,Height-1,Width-1])

        #out : torch.Size([batch_size,1,Height-1,Width-1])
        #cam_logit : torch.Size([batch_size,2])
        #heatmap : torch.Size([batch_size,1,Height,Width])
        return out, cam_logit, heatmap
