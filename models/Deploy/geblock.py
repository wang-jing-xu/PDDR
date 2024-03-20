import torch.nn as nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3, relu=True, stride=2, padding=1):
        super(Downblock, self).__init__()

        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu
        self.stride = stride

    def forward(self, x):
        if x.shape[-1]//self.stride==1:
            return x
        x = self.dwconv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x


class GEBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, spatial=3, extent=8, extra_params=True, mlp=False, dropRate=0.0):
        # If extent is zero, assuming global.
        super(GEBlock, self).__init__()
        self.extent = extent

        if extra_params is True:
            if extent == 0:
                # Global DW Conv + BN
                self.downop = Downblock(out_planes, relu=False, kernel_size=spatial, stride=1, padding=0)
            elif extent == 2:
                self.downop = Downblock(out_planes, relu=False)

            elif extent == 4:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))
            elif extent == 8:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))

            else:

                raise NotImplementedError('Extent must be 0,2,4 or 8 for now')
        else:
            if extent == 0:
                self.downop = nn.AdaptiveAvgPool2d(1)

            else:
                self.downop = nn.AdaptiveAvgPool2d(spatial // extent)

        if mlp is True:
            self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16, kernel_size=1, padding=0, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False),
                                     )
        else:
            self.mlp = Identity()

    def forward(self, x):
        # Assuming squares because lazy.
        shape_in = x.shape[-1]

        # Down, up, sigmoid
        map = self.downop(x)
        map = self.mlp(map)
        map = F.interpolate(map, shape_in)
        map = torch.sigmoid(map)
        return x * map
