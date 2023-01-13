import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up_cat3(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up_cat3, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 3, in_ch // 3, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class Upself(Up):
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch, sig):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.Sigmoid()
        self.sig = sig

    def forward(self, x):
        x = self.conv(x)
        if self.sig:
            x = self.act(x)
        return x


class Outconv2(nn.Module):
    def __init__(self, in_ch, out_ch, sig):
        super(Outconv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(in_ch),
            nn.InstanceNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.act = nn.Sigmoid()
        self.sig = sig

    def forward(self, x):
        x = self.conv(x)
        if self.sig:
            x = self.act(x)
        return x


class CoDetectionCNN(nn.Module):
    def __init__(self, n_channels, n_classes, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3_1 = Up(256, 64)
        self.up3_2 = Up(256, 64)
        self.up4_1 = Up(128, 32)
        self.up4_2 = Up(128, 32)
        
        self.up1_diff = Up(1024, 256)
        self.up2_diff = Up(512, 128)
        self.up3_diff = Up_cat3(128*3, 64)
        self.up4_diff = Up_cat3(64*3, 32)
        
        self.out_1 = Outconv2(32, n_classes, sig=sig)
        self.out_2 = Outconv2(32, n_classes, sig=sig)
        self.out_diff = Outconv2(32, n_classes, sig=sig)

    def forward(self, x, diff=True):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1::, :, :]
        
        # encoder
        down0_1 = self.inc(x1)   # B, 64, 512, 512
        down0_2 = self.inc(x2)   # B, 64, 512, 512
        down1_1 = self.down1(down0_1)   # B, 128, 256, 256
        down1_2 = self.down1(down0_2)   # B, 128, 256, 256
        x_cat = torch.cat([down1_1, down1_2], dim=1)   # B, 256, 256, 256
        down2 = self.down2(x_cat)   # B, 256, 128, 128
        down3 = self.down3(down2)   # B, 512, 64, 64
        down4 = self.down4(down3)   # B, 512, 32, 32
        
        # decoder
        up1 = self.up1(down4, down3)   # B, 256, 64, 64
        up2 = self.up2(up1, down2)   # B, 128, 128, 128
        up3_1 = self.up3_1(up2, down1_1)   # B, 64, 256, 256
        up3_2 = self.up3_2(up2, down1_2)   # B, 64, 256, 256
        up4_1 = self.up4_1(up3_1, down0_1)   # B, 32, 512, 512
        up4_2 = self.up4_2(up3_2, down0_2)   # B, 32, 512, 512
        
        # output
        out_1 = self.out_1(up4_1)   # B, 2, 512, 512
        out_2 = self.out_2(up4_2)   # B, 2, 512, 512
        
        if diff:
            # decoder2
            up1_diff = self.up1_diff(down4, down3)
            up2_diff = self.up2_diff(up1_diff, down2)
            up3_diff = self.up3_diff(up2_diff, down1_1, down1_2)
            up4_diff = self.up4_diff(up3_diff, down0_1, down0_2)
            out_diff = self.out_diff(up4_diff)
            return out_1, out_2, out_diff
        else:
            return out_1, out_2

class CoDetectionCNN_affs(nn.Module):
    def __init__(self, n_channels, n_classes, n_affs, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3_1 = Up(256, 64)
        self.up3_2 = Up(256, 64)
        self.up4_1 = Up(128, 32)
        self.up4_2 = Up(128, 32)
        
        self.up1_diff = Up(1024, 256)
        self.up2_diff = Up(512, 128)
        self.up3_diff = Up_cat3(128*3, 64)
        self.up4_diff = Up_cat3(64*3, 32)
        
        self.out_1 = Outconv2(32, n_classes, sig=sig)
        self.out_2 = Outconv2(32, n_classes, sig=sig)
        self.out_diff = Outconv2(32, n_affs, sig=True)

    def forward(self, x, diff=True):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1::, :, :]
        
        # encoder
        down0_1 = self.inc(x1)   # B, 64, 512, 512
        down0_2 = self.inc(x2)   # B, 64, 512, 512
        down1_1 = self.down1(down0_1)   # B, 128, 256, 256
        down1_2 = self.down1(down0_2)   # B, 128, 256, 256
        x_cat = torch.cat([down1_1, down1_2], dim=1)   # B, 256, 256, 256
        down2 = self.down2(x_cat)   # B, 256, 128, 128
        down3 = self.down3(down2)   # B, 512, 64, 64
        down4 = self.down4(down3)   # B, 512, 32, 32
        
        # decoder
        up1 = self.up1(down4, down3)   # B, 256, 64, 64
        up2 = self.up2(up1, down2)   # B, 128, 128, 128
        up3_1 = self.up3_1(up2, down1_1)   # B, 64, 256, 256
        up3_2 = self.up3_2(up2, down1_2)   # B, 64, 256, 256
        up4_1 = self.up4_1(up3_1, down0_1)   # B, 32, 512, 512
        up4_2 = self.up4_2(up3_2, down0_2)   # B, 32, 512, 512
        
        # output
        out_1 = self.out_1(up4_1)   # B, 2, 512, 512
        out_2 = self.out_2(up4_2)   # B, 2, 512, 512
        
        if diff:
            # decoder2
            up1_diff = self.up1_diff(down4, down3)
            up2_diff = self.up2_diff(up1_diff, down2)
            up3_diff = self.up3_diff(up2_diff, down1_1, down1_2)
            up4_diff = self.up4_diff(up3_diff, down0_1, down0_2)
            out_diff = self.out_diff(up4_diff)
            return out_1, out_2, out_diff
        else:
            return out_1, out_2


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    x = torch.rand((1, 2, 512, 512))
    model = CoDetectionCNN(n_channels=1, n_classes=2, sig=False)
    out = model(x)
    for i in range(len(out)):
        print(out[i].shape)
    
    macs, params = get_model_complexity_info(model, (2, 512, 512), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))