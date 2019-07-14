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


def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            # this is the multiplier that are used for equalized learning rate
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            kernelSize (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious, nChannels,
                                            kernelSize, padding=padding,
                                            bias=bias),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), **kwargs)


class EqualizedBiasMixin(nn.Module):

    def __init__(self, num_channels, lrmul=1.):
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.lrmul = lrmul

    def forward(self, x):
        dim = x.dim()
        if dim == 2:
            return x + self.bias * self.lrmul
        else:
            # channel wise
            x = x + self.bias.view(1, -1, 1, 1) * self.lrmul
            return x


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
            noise = torch.randn(s[0], 1, s[2], s[3], device=x.divice, dtype=x.dtype)
        x = x + self.weight.view(1, -1, 1, 1) * noise

        return x


class StyleMixin(nn.Module):
    """
    Style modulation.
    reference: style_mod in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """
    def __init__(self,
                 dlatent,          # Disentangled latent (W) dimensionality
                 num_output_channels,
                 use_wscale        # use equalized learning rate?
                 ):
        super(StyleMixin, self).__init__()
        if use_wscale:
            self.linear = EqualizedLinear(dlatent, num_output_channels)
        else:
            self.linear = nn.Linear(dlatent, num_output_channels, bias=True)


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
        shape = [-1, 2, x.size()[1]] + [1] * (x.dim() - 2)
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
                 dlatent_size=512,        # Disentangled latent (W) dimensionality,
                 use_wscale=True,         # Enable equalized learning rate?
                 use_pixel_norm=False,    # Enable pixel-wise feature vector normalization?
                 use_instance_norm=True,
                 use_noise=True,
                 use_styles=True,
                 nonlinearity='lrelu',
                 ):
        super(LayerEpilogue, self).__init__()

        act = {
               'relu': torch.relu,
               'lrelu': nn.LeakyReLU(negative_slope=0.2)
               }[nonlinearity]

        layers = []
        if use_noise:
            layers.append(('noise', NoiseMixin(num_channels)))
        # lrmul = 1.0 here?
        layers.append(('bias', EqualizedBiasMixin(num_channels)))
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
                 nf,
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
        self.nf = nf

        if const_input_layer:
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
        self.epi1 = LayerEpilogue(nf,
                                  dlatent_size,
                                  use_wscale,
                                  use_pixel_norm,
                                  use_instance_norm,
                                  use_noise,
                                  use_styles,
                                  nonlinearity
                                  )
        self.conv = EqualizedConv2d(nf, nf, 3)

        self.epi2 = LayerEpilogue(nf,
                                  dlatent_size,
                                  use_wscale,
                                  use_pixel_norm,
                                  use_instance_norm,
                                  use_styles,
                                  nonlinearity)

    def forward(self, dlatents):
        # note dlatents is broadcast one
        dlatents_0 = dlatents[:, 0]
        dlatents_1 = dlatents[:, 1]
        batch_size = dlatents.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)

        x = self.epi1(x, dlatents_0)
        x = self.conv(x)
        x = self.epi2(x, dlatents_1)
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
                 ):
        super(LaterBlock, self).__init__()
        if blur_filter:
            self.
        self.epi1
