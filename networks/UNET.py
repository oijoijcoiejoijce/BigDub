import os
import torch
import torch.nn as nn
import functools


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # use_norm = False
        use_norm = True

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)

        if use_norm: downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        if use_norm: upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            if use_norm: down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# with bilinear upsampling
class UnetSkipConnectionBlock_BU(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock_BU, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # use_norm = False
        use_norm = True

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)

        if use_norm: downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        if use_norm: upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inner_nc * 2, inner_nc * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(inner_nc * 2, inner_nc * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            )

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]
            model = down + up
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            if use_norm: down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# with bilinear upsampling
class UnetSkipConnectionBlock_ADAIN(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, cond_nc=512):
        super(UnetSkipConnectionBlock_ADAIN, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # use_norm = False
        use_norm = True

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)

        if use_norm: downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        if use_norm: upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inner_nc * 2, inner_nc * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(inner_nc * 2, inner_nc * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=1, stride=1, padding=0, bias=use_bias),
            )

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]
            model = down + up
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            if use_norm: down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.ModuleList(model)
        self.cond_layer = nn.Sequential(
            nn.Linear(cond_nc, outer_nc, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Linear(outer_nc, outer_nc * 2, bias=False)
        )

    def forward(self, x, cond=None):
        if self.outermost:
            for layer in self.model:
                if isinstance(layer, UnetSkipConnectionBlock_ADAIN):
                    x = layer(x, cond)
                else:
                    x = layer(x)
            return x
        else:
            y = x.clone()
            for layer in self.model:
                if isinstance(layer, UnetSkipConnectionBlock_ADAIN):
                    y = layer(y, cond)
                else:
                    y = layer(y)

            cond_mapped = self.cond_layer(cond)
            mu, sigma = torch.chunk(cond_mapped, 2, dim=-1)
            sigma = sigma + 1
            y = (y * sigma[..., None, None]) + mu[..., None, None]
            return torch.cat([x, y], 1)


# dilated convs, without downsampling
class UnetSkipConnectionBlock_DC(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, dilation=1):
        super(UnetSkipConnectionBlock_DC, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # use_norm = False
        use_norm = True

        # downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, dilation=dilation, padding=1, bias=use_bias)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, dilation=dilation, padding=1 * dilation,
                             bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        if use_norm: downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        if use_norm: upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv = nn.Sequential(
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, dilation=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, dilation=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]
            model = down + up
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Sequential(
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, dilation=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

            if use_norm: down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            if use_norm:
                up = [uprelu, upconv, upnorm]
            else:
                up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetRenderer(nn.Module):
    def __init__(self, renderer, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetRenderer, self).__init__()

        self.renderer = renderer
        # construct unet structure
        if renderer == 'UNET_8_level_BU':
            print('>>>> UNET_8_level_BU <<<<')
            num_downs = 8
            unet_block = UnetSkipConnectionBlock_BU(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock_BU(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                        norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock_BU(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=norm_layer)
        elif renderer == 'UNET_6_level_BU':
            print('>>>> UNET_6_level_BU <<<<')
            num_downs = 6
            unet_block = UnetSkipConnectionBlock_BU(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock_BU(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                        norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock_BU(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=norm_layer)
        elif renderer == 'UNET_5_level_BU':
            print('>>>> UNET_5_level_BU <<<<')
            num_downs = 5
            unet_block = UnetSkipConnectionBlock_BU(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock_BU(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                        norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock_BU(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=norm_layer)
        elif renderer == 'UNET_3_level_BU':
            print('>>>> UNET_3_level_BU <<<<')
            unet_block = UnetSkipConnectionBlock_BU(ngf * 2, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock_BU(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_BU(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=norm_layer)

        elif renderer == 'UNET_8_level':
            print('>>>> UNET_8_level <<<<')
            num_downs = 8
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                 innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                 outermost=True, norm_layer=norm_layer)
        elif renderer == 'UNET_6_level':
            print('>>>> UNET_6_level <<<<')
            num_downs = 6
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                 innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                 outermost=True, norm_layer=norm_layer)
        elif renderer == 'UNET_5_level':
            print('>>>> UNET_5_level <<<<')
            num_downs = 5
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                 innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                 outermost=True, norm_layer=norm_layer)
        elif renderer == 'UNET_3_level':
            print('>>>> UNET_3_level <<<<')
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                 innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                 outermost=True, norm_layer=norm_layer)

        elif renderer == 'UNET_5_level_DC':
            print('>>>> UNET_5_level_DC <<<<')
            num_downs = 5
            dilation = 1
            unet_block = UnetSkipConnectionBlock_DC(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=norm_layer, innermost=True, dilation=dilation)
            for i in range(num_downs - 5):
                dilation *= 2
                unet_block = UnetSkipConnectionBlock_DC(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                        norm_layer=norm_layer, use_dropout=use_dropout,
                                                        dilation=dilation)
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, dilation=dilation)
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, dilation=dilation)
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, dilation=dilation)
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=norm_layer, dilation=dilation)
        elif renderer == 'UNET_3_level_DC':
            print('>>>> UNET_3_level_DC <<<<')
            dilation = 1
            unet_block = UnetSkipConnectionBlock_DC(ngf * 2, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=norm_layer, innermost=True, dilation=dilation)
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, dilation=dilation)
            dilation *= 2
            unet_block = UnetSkipConnectionBlock_DC(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=norm_layer, dilation=dilation)

        elif renderer == 'UNET_5_level_ADAIN':
            print('>>>> UNET_5_level_ADAIN <<<<')
            num_downs = 5
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                    norm_layer=nn.InstanceNorm2d, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                        norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=nn.InstanceNorm2d)
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                    norm_layer=nn.InstanceNorm2d)
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                    norm_layer=nn.InstanceNorm2d)
            unet_block = UnetSkipConnectionBlock_ADAIN(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                    outermost=True, norm_layer=nn.InstanceNorm2d)

        elif renderer == 'UNET_8_level_ADAIN':
            print('>>>> UNET_8_level_ADAIN <<<<')
            num_downs = 8
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                 innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_ADAIN(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock_ADAIN(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                 outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, features, cond=None):
        unet_input = features

        if 'ADAIN' in self.renderer:
            return self.model(unet_input, cond)
        return self.model(unet_input)


