import math
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import prod


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


class NormalizationLayer2(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class Upscale2d(nn.Module):

    def __init__(self, factor=2):
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        factor = self.factor
        if factor == 1:
            return x
        s = x.size()
        x = x.view(-1, s[1], s[2], 1, s[3], 1)
        x = x.expand(-1, s[1], s[2], factor, s[3], factor)
        x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x

class Upscale2d2(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor
    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class Blur2d(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(Blur2d, self).__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


def getLayerNormalizationFactor(x, gain):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return gain * math.sqrt(1.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 use_wscale=True,
                 lrmul=1.0,
                 bias=True,
                 gain=np.sqrt(2)):
        r"""
        use_wscale (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        init_bias_to_zero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = use_wscale

        if bias:
            # size(0) is num_out_channels
            self.bias = torch.nn.Parameter(torch.zeros(self.module.weight.size(0)))
            self.bias_mul = 1.0
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrmul
            # this is the multiplier that are used for equalized learning rate
            self.weight_mul = getLayerNormalizationFactor(self.module, gain=gain) * lrmul
            self.bias_mul = lrmul

    def forward(self, x):

        # hack hack. module's bias is always false
        x = self.module(x)
        if self.equalized:
            # this is different from the tf implementation!
            x *= self.weight_mul
        # add on bias
        if self.bias is not None:
            if x.dim() == 2:
                x = x + self.bias.view(1, -1) * self.bias_mul
            else:
                x = x + self.bias.view(1, -1, 1, 1) * self.bias_mul
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 num_input_channels,
                 num_output_channels,
                 kernel_size,
                 padding=0,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            num_input_channels (int): number of channels in the previous layer
            num_output_channels (int): number of channels of the current layer
            kernel_size (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """

        # always set bias to False
        # and apply bias manually
        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(num_input_channels, num_output_channels,
                                            kernel_size, padding=padding,
                                            bias=False),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 num_input_channels,
                 num_output_channels,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            num_input_channels (int): number of channels in the previous layer
            num_output_channels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(num_input_channels, num_output_channels,
                                            bias=False), **kwargs)


class SmoothUpsample(nn.Module):
    """
    https://arxiv.org/pdf/1904.11486.pdf
    'Making Convolutional Networks Shift-Invariant Again'
    # this is in the tf implementation too
    """
    def __init__(self,
                 num_input_channels,
                 num_output_channels,
                 kernel_size,
                 padding=0,
                 bias=True):
        super(SmoothUpsample, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_output_channels,
                                               num_input_channels,
                                               kernel_size,
                                               kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_output_channels))
        else:
            self.bias = None
        self.padding = padding

    def forward(self, x):
        # change to in_channels, out_channels, kernel_size, kernel_size
        weight = self.weight.permute([1, 0, 2, 3])
        weight = F.pad(weight, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:]
                  + weight[:, :, :-1, 1:]
                  + weight[:, :, 1:, :-1]
                  + weight[:, :, :-1, :-1]
                 )
        x = F.conv_transpose2d(x,
                               weight,
                               self.bias, # note if bias set to False, this will be None
                               stride=2,
                               padding=self.padding)
        return x

# TODO: this needs to be better wrappered by ConstrainedLayer for bias
class EqulizedSmoothUpsample(ConstrainedLayer):
    def __init__(self,
                 num_input_channels,
                 num_output_channels,
                 kernel_size,
                 padding=0,
                 **kwargs):
        ConstrainedLayer.__init__(self, SmoothUpsample(num_input_channels,
                                                       num_output_channels,
                                                       kernel_size=kernel_size,
                                                       padding=padding,
                                                       bias=False), **kwargs)


class Upscale2dConv2d(nn.Module):

    def __init__(self,
                 res,              # this is used  for determin the fused_scale
                 num_input_channels,
                 num_output_channels,
                 kernel_size,
                 use_wscale,
                 fused_scale='auto',
                 **kwargs):
        super(Upscale2dConv2d, self).__init__()
        # kernel_size assert (from official tf implementation):
        # this is due to the fact that the input size is always even
        # and use kernel_size // 2 ensures 'same' padding from tf
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.fused_scale = fused_scale
        self.upscale = Upscale2d2()

        if self.fused_scale == 'auto':
            self.fused_scale = (2 ** res) >= 128

        if not self.fused_scale:

            self.conv = EqualizedConv2d(num_input_channels,
                                    num_output_channels,
                                    kernel_size,
                                    padding=kernel_size // 2,
                                    use_wscale=use_wscale,
                                    )
        else:
            self.conv = EqulizedSmoothUpsample(num_input_channels,
                                               num_output_channels,
                                               kernel_size,
                                               padding=(kernel_size - 1) // 2,
                                               use_wscale=use_wscale,
                                               )

    def forward(self, x):
        if not self.fused_scale:
            return self.conv(self.upscale(x))
        else:
            return self.conv(x)


class NoiseMixin(nn.Module):
    """
    Add noise with channel wise scaling factor
    reference: apply_noise in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """
    def __init__(self, num_channels):
        super(NoiseMixin, self).__init__()
        # initialize with 0's
        # per-channel scaling factor
        # 'B' in the paper
        # use weight to match the tf implementation
        self.weight = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, noise=None):
        # NCHW
        assert len(x.size()) == 4
        s = x.size()
        if noise is None:
            noise = torch.randn(s[0], 1, s[2], s[3], device=x.device, dtype=x.dtype)
        x = x + self.weight.view(1, -1, 1, 1) * noise

        return x


class StyleMixin(nn.Module):
    """
    Style modulation.
    reference: style_mod in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """
    def __init__(self,
                 dlatent_size,          # Disentangled latent (W) dimensionality
                 num_channels,
                 use_wscale        # use equalized learning rate?
                 ):
        super(StyleMixin, self).__init__()
        # gain is 1.0 here
        self.linear = EqualizedLinear(dlatent_size, num_channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, w):
        # x is instance normalized
        # w is mapped latent
        style = self.linear(w)
        # style's shape (N, 2 * 512)
        # reshape to (y_s, y_b)
        # style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        # NCHW
        # according to the paper, shape of style (y) would be (N, 2, 512, 1, 1)
        # so shape of y_s is (N, 512, 1, 1)
        # channel-wise, y_s is just a scalar
        shape = [-1, 2, x.size(1)] + [1] * (x.dim() - 2)
        style = style.view(shape)
        return x * (style[:, 0] + 1.) + style[:, 1]


class LayerEpilogue(nn.Module):
    """
    Things to do at the end of each layer
    1. mixin scaled noise
    2. mixin style with AdaIN
    """
    def __init__(self,
                 num_channels,
                 dlatent_size,        # Disentangled latent (W) dimensionality,
                 use_wscale,         # Enable equalized learning rate?
                 use_pixel_norm,    # Enable pixel-wise feature vector normalization?
                 use_instance_norm,
                 use_noise,
                 use_styles,
                 nonlinearity,
                 ):
        super(LayerEpilogue, self).__init__()

        act = {
               'relu': torch.relu,
               'lrelu': nn.LeakyReLU(negative_slope=0.2)
               }[nonlinearity]

        layers = []
        if use_noise:
            layers.append(('noise', NoiseMixin(num_channels)))
        layers.append(('act', act))

        # to follow the tf implementation
        if use_pixel_norm:
            layers.append(('pixel_norm', NormalizationLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(num_channels)))
        # now we need to mixin styles
        self.pre_style_op = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMixin(dlatent_size,
                                        num_channels,
                                        use_wscale=use_wscale)
    def forward(self, x, dlatent):
        # dlatent is w
        x = self.pre_style_op(x)
        if self.style_mod:
            x = self.style_mod(x, dlatent)
        return x


class EarlyBlock(nn.Module):
    """
    The first block for 4x4 resolution
    """
    def __init__(self,
                 in_channels,
                 dlatent_size,
                 const_input_layer,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles,
                 nonlinearity
                 ):
        super(EarlyBlock, self).__init__()
        self.const_input_layer = const_input_layer
        self.in_channels = in_channels

        if const_input_layer:
            self.const = nn.Parameter(torch.ones(1, in_channels, 4, 4))
            self.bias = nn.Parameter(torch.ones(in_channels))
        else:
            self.dense = EqualizedLinear(dlatent_size, in_channels * 16, equalized=use_wscale)

        self.epi0 = LayerEpilogue(num_channels=in_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_noise=use_noise,
                                  use_pixel_norm=use_pixel_norm,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity
                                  )
        # kernel size must be 3 or other odd numbers
        # so that we have 'same' padding
        self.conv = EqualizedConv2d(num_input_channels=in_channels,
                                    num_output_channels=in_channels,
                                    kernel_size=3,
                                    padding=3//2)

        self.epi1 = LayerEpilogue(num_channels=in_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_noise=use_noise,
                                  use_pixel_norm=use_pixel_norm,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity
                                  )

    def forward(self, dlatents):
        # note dlatents is broadcast one
        dlatents_0 = dlatents[:, 0]
        dlatents_1 = dlatents[:, 1]
        batch_size = dlatents.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_0).view(batch_size, self.in_channels, 4, 4)

        x = self.epi0(x, dlatents_0)
        x = self.conv(x)
        x = self.epi1(x, dlatents_1)
        return x


class LaterBlock(nn.Module):
    """
    The following blocks for res 8x8...etc.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles,
                 nonlinearity,
                 blur_filter,
                 res,
                 ):
        super(LaterBlock, self).__init__()

        # res = log2(H), H is 4, 8, 16, 32 ... 1024

        assert isinstance(res, int) and (2 <= res <= 10)

        self.res = res

        if blur_filter:
            self.blur = Blur2d(blur_filter)
            #blur = Blur2d(blur_filter)
        else:
            self.blur = None

        # name 'conv0_up' is used in tf implementation
        self.conv0_up = Upscale2dConv2d(res=res,
                                        num_input_channels=in_channels,
                                        num_output_channels=out_channels,
                                        kernel_size=3,
                                        use_wscale=use_wscale)
       # self.conv0_up = Upscale2dConv2d2(
       #     input_channels=in_channels,
       #     output_channels=out_channels,
       #     kernel_size=3,
       #     gain=np.sqrt(2),
       #     use_wscale=use_wscale,
       #     intermediate=blur,
       #     upscale=True
       # )

        self.epi0 = LayerEpilogue(num_channels=out_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_pixel_norm=use_pixel_norm,
                                  use_noise=use_noise,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity)

        # name 'conv1' is used in tf implementation
        # kernel size must be 3 or other odd numbers
        # so that we have 'same' padding
        # no upsclaing
        self.conv1 = EqualizedConv2d(num_input_channels=out_channels,
                                    num_output_channels=out_channels,
                                    kernel_size=3,
                                    padding=3//2)

        self.epi1 = LayerEpilogue(num_channels=out_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_pixel_norm=use_pixel_norm,
                                  use_noise=use_noise,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity)


    def forward(self, x, dlatent):

        x = self.conv0_up(x)
        if self.blur is not None:
            x = self.blur(x)
        x = self.epi0(x, dlatent[:, 0])
        x = self.conv1(x)
        x = self.epi1(x, dlatent[:, 1])
        return x

