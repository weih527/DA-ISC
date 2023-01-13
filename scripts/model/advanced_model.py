# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid

class Double_conv(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            # nn.InstanceNorm2d(out_ch),
            nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-5,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-5,affine=False),
            # nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x


class Conv_up(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


class Conv_up_nl(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up_nl, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1):
        x1 = self.interp(x1)
        x1 = self.conv(x1)
        return x1


def extract_img(size, in_tensor):
    """
    Args:
        size(int) : size of cut
        in_tensor(tensor) : tensor to be cut
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1 - size) / 2):int((dim1 + size) / 2),
                int((dim2 - size) / 2):int((dim2 + size) / 2)]
    return in_tensor


class CleanU_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CleanU_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 64)
        self.Conv_down2 = Conv_down(64, 128)
        self.Conv_down3 = Conv_down(128, 256)
        self.Conv_down4 = Conv_down(256, 512)
        self.Conv_down5 = Conv_down(512, 1024)

        self.Conv_up1 = Conv_up(1024, 512)
        self.Conv_up2 = Conv_up(512, 256)
        self.Conv_up3 = Conv_up(256, 128)
        self.Conv_up4 = Conv_up(128, 64)
        self.Conv_out = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)

    def forward(self, x):
        x, conv1 = self.Conv_down1(x)
        # print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        # print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        # print("dConv3 => down3|", x.shape)
        x, conv4 = self.Conv_down4(x)
        # print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        # print("dConv5|", x.shape)
        x = self.Conv_up1(x, conv4)
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        # print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, conv2)
        # print("up3 => uConv3|", x.shape)
        x = self.Conv_up4(x, conv1)
        feature = x
        output = self.Conv_out(x)
        return feature, output

class DoubleU_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleU_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 64)
        self.Conv_down2 = Conv_down(64, 128)
        self.Conv_down3 = Conv_down(128, 256)
        self.Conv_down4 = Conv_down(256, 512)
        self.Conv_down5 = Conv_down(512, 1024)

        self.Conv_up1 = Conv_up(1024, 512)
        self.Conv_up2 = Conv_up(512, 256)
        self.Conv_up3 = Conv_up(256, 128)
        self.Conv_out1 = nn.Conv2d(128,out_channels,1,padding = 0,stride = 1)
        self.Conv_up4 = Conv_up(128, 64)
        self.Conv_out2 = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)

    def forward(self, x):
        x, conv1 = self.Conv_down1(x)
        # print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        # print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        # print("dConv3 => down3|", x.shape)
        x, conv4 = self.Conv_down4(x)
        # print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        # print("dConv5|", x.shape)
        x = self.Conv_up1(x, conv4)
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        # print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, conv2)
        # print("up3 => uConv3|", x.shape)
        x1 = self.Conv_out1(x)
        x = self.Conv_up4(x, conv1)
        x2 = self.Conv_out2(x)
        return x1, x2


class U_NetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(U_NetEncoder, self).__init__()
        self.in_channels = in_channels
        self.Conv_down1 = Conv_down(in_channels, 64)
        self.Conv_down2 = Conv_down(64, 128)
        self.Conv_down3 = Conv_down(128, 256)
        self.Conv_down4 = Conv_down(256, 512)
        self.Conv_down5 = Conv_down(512, 1024)

    def forward(self, x):
        encoder_outputs = []
        x, conv1 = self.Conv_down1(x)
        encoder_outputs.append(conv1)
        x, conv2 = self.Conv_down2(x)
        encoder_outputs.append(conv2)
        x, conv3 = self.Conv_down3(x)
        encoder_outputs.append(conv3)
        x, conv4 = self.Conv_down4(x)
        _, x = self.Conv_down5(x)
        encoder_outputs.append(conv4)
        return x, encoder_outputs


class Domain_Decoder(nn.Module):
    def __init__(self, out_channels=2):
        super(Domain_Decoder, self).__init__()
        self.out_channels = out_channels
        self.Conv_up1 = Conv_up(1024, 512)
        self.Conv_up2 = Conv_up(512, 256)
        self.Conv_up3 = Conv_up(256, 128)
        self.Conv_up4 = Conv_up(128, 64)
        self.Conv_out = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)

    def forward(self, x, encoder_outputs):
        encoder_outputs.reverse()
        # print("dConv5|", x.shape)
        x = self.Conv_up1(x, encoder_outputs[0])
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, encoder_outputs[1])
        # print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, encoder_outputs[2])
        # print("up3 => uConv3|", x.shape)
        x = self.Conv_up4(x, encoder_outputs[3])
        feature = x
        output = self.Conv_out(x)

        return feature, output


class Rec_Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(Rec_Decoder, self).__init__()
        self.out_channels = out_channels
        self.Conv_up1 = Conv_up_nl(1024, 512)
        self.Conv_up2 = Conv_up_nl(512, 256)
        self.Conv_up3 = Conv_up_nl(256, 128)
        self.Conv_up4 = Conv_up_nl(128, 64)
        self.Conv_out = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)

    def forward(self, x):
        # print("dConv5|", x.shape)
        x = self.Conv_up1(x)
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x)
        # print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x)
        # print("up3 => uConv3|", x.shape)
        x = self.Conv_up4(x)
        output = self.Conv_out(x)

        return output

class source2targetNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(source2targetNet, self).__init__()

        self.encoder = U_NetEncoder(in_channels)
        self.domain_decoder = Domain_Decoder(out_channels)
        self.rec_decoder = Rec_Decoder(in_channels)

    def forward(self, inputs):
        x, encoder_outputs = self.encoder(inputs)
        feature, pred = self.domain_decoder(x, encoder_outputs)
        recimg = self.rec_decoder(x)

        return recimg, feature, pred

    def get_target_segmentation_net(self):
        return unet_from_encoder_decoder(self.encoder, self.domain_decoder)

class source2targetNet_seg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(source2targetNet_seg, self).__init__()

        self.encoder = U_NetEncoder(in_channels)
        self.domain_decoder = Domain_Decoder(out_channels)

    def forward(self, inputs):
        x, encoder_outputs = self.encoder(inputs)
        feature, pred = self.domain_decoder(x, encoder_outputs)

        return feature, pred

class UNet2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # contractive path
        self.encoder = U_NetEncoder(in_channels)
        # expansive path
        self.decoder = Domain_Decoder(out_channels)

    def forward(self, inputs):
        x, encoder_outputs = self.encoder(inputs)

        feature, x = self.decoder(x, encoder_outputs)

        return feature, x


def unet_from_encoder_decoder(encoder, decoder):
    net = UNet2D(in_channels=encoder.in_channels, out_channels=decoder.out_channels)

    new_encoder_dict = net.encoder.state_dict()
    new_decoder_dict = net.decoder.state_dict()

    encoder_dict = encoder.state_dict()
    decoder_dict = decoder.state_dict()

    for k, v in encoder_dict.items():
        if k in new_encoder_dict:
            new_encoder_dict[k] = v
        else:
            print("val model encoder parameter copy error!!!")
    net.encoder.load_state_dict(new_encoder_dict)

    for k, v in decoder_dict.items():
        if k in new_decoder_dict:
            new_decoder_dict[k] = v
        else:
            print("val model decoder parameter copy error!!!")
    net.decoder.load_state_dict(new_decoder_dict)

    return net

class DomainDiscriminator(nn.Module):
    def __init__(self, input_channels,input_size,num_classes,fc_classifier = 3):
        super(DomainDiscriminator, self).__init__()
        self.fc_classifier = fc_classifier
        self.fc_channels = [288, 144, 2]
        self.conv_channels = [48, 48, 48]
        self.input_size = input_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_channels
        data_size = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                 nn.ReLU())
            self.conv_features.add_module('conv%d' % (i + 1), conv)
            in_channels = out_channels
            data_size /= 2

        # full connections
        in_channels = self.conv_channels[-1]*data_size*data_size
        for i, out_channels in enumerate(self.fc_channels):
            if i == fc_classifier - 1:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels))
            else:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels),
                                   nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                   nn.ReLU())
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.gpN = nn.GroupNorm(num_groups=4, num_channels=num_classes, eps=0, affine=False)
        # nn.InstanceNorm2d(out_ch),
        # nn.Dropout(p=0.2),

    # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
    # self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        for i in range(len(self.conv_channels)):
            x = getattr(self.conv_features, 'conv%d' % (i + 1))(x)
            x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.size(0), -1)
        for i in range(self.fc_classifier):
            x = getattr(self.fc_features, 'linear%d' % (i + 1))(x)

        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    input = np.random.random((2,1,512,512)).astype(np.float32)
    x = torch.tensor(input).to('cuda:0')

    model = source2targetNet_seg(in_channels=1, out_channels=2).cuda()
    # recimg, feature, pred = model(x)
    # print(recimg.shape)
    feature, pred = model(x)
    print(feature.shape)
    print(pred.shape)
    
    macs, params = get_model_complexity_info(model, (1, 512, 512), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))