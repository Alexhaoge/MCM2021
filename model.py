import torch
from torch import nn
import torch.nn.functional as F


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Stem_v4(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """
    def __init__(self):
        super(Stem_v4, self).__init__()
        self.step1 = nn.Sequential(
            BN_Conv2d(3, 32, 3, 2, 0, bias=False),
            BN_Conv2d(32, 32, 3, 1, 0, bias=False),
            BN_Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step3_2 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        # print(tmp1.shape)
        # print(tmp2.shape)
        out = torch.cat((tmp1, tmp2), 1)
        return out


class Inception_A(nn.Module):
    """
    Inception-A block for Inception-v4 net
    """
    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3, 3, 1, 1, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n3, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n3, b4_n3, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2 nets
    """
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = BN_Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, k, 1, 1, 0, bias=False),
            BN_Conv2d(k, l, 3, 1, 1, bias=False),
            BN_Conv2d(l, m, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Inception_B(nn.Module):
    """
    Inception-B block for Inception-v4 net
    """
    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1,
                 b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """
    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False)
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Inception_C(nn.Module):
    """
    Inception-C block for Inception-v4 net
    """
    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1,
                 b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3_1 = BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False)
        self.branch3_1x3 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_3x1 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
        )
        self.branch4_1x3 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_3x1 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return torch.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)


class Inception(nn.Module):
    """
    implementation of Inception-v4
    """
    def __init__(self, num_classes):
        super(Inception, self).__init__()
        self.stem = Stem_v4()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        self.fc = nn.Linear(1536, num_classes)

    def __make_inception_A(self):
        layers = []
        for _ in range(4):
            layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        return Reduction_A(384, 192, 224, 256, 384)  # 1024

    def __make_inception_B(self):
        layers = []
        for _ in range(7):
            layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                    192, 192, 224, 224, 256))   # 1024
        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)  # 1536

    def __make_inception_C(self):
        layers = []
        for _ in range(3):
            layers.append(Inception_C(1536, 256, 256,
                                        384, 256, 384, 448, 512, 256))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        out = F.avg_pool2d(out, 8)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return F.softmax(out, dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        """
        Focal loss function, -α(1-yi)**γ *ce_loss(xi,yi)      
        :param alpha:   α, class weight
        :param gamma:   γ
        :param size_average:
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        assert alpha < 1
        self.alpha = torch.zeros(2)
        self.alpha[0] += alpha
        self.alpha[1:] += (1-alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        """     
        :param preds:   Predict. size: [B,C]      
        :param labels:  Actual. size: [B]        
        :return:
        """
        assert preds.dim() == 2 and labels.dim() == 1
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = torch.log(preds)
        preds_softmax = preds.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        loss = loss.mean() if self.size_average else loss.sum()
        return loss
