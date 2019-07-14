import torch
import torch.nn as nn
from collections import OrderedDict
from networks.custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d

class MappingNet(nn.Module):
    """
    A mapping network f implemented using an 8-layer MLP
    """
    def __init__(self,
                 num_layers,
                 latent_dim,
                 dlatent_broadcast=None, # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
                 normalize_latents=True,
                 nonlinearity='lrelu',
                 maping_lrmul=0.01):

        super(MappingNet, self).__init__()

        act = {
            'relu': torch.relu,
            'lrelu': nn.LeakyReLU(negative_slope=0.2)
        }[nonlinearity]

        layers = []
        if normalize_latents:
            layers.append(('pixel_norm', NormalizationLayer()))
        for i in range(num_layers):
            layers.append(('fc{}'.format(i), EqualizedLinear(latent_dim,
                                                             latent_dim,
                                                             lrMul=maping_lrmul)))
            layers.append(('fc{}_act'.format(i), act))

        self.layers = nn.Sequential(OrderedDict(layers))
        self.dlatent_broadcast = dlatent_broadcast

    def forward(self, x):
        # N x 512
        w = self.layers(x)
        if self.dlatent_broadcast is not None:
            # broadcast
            # tf.tile in the official tf implementation:
            # w = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])
            w = w.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return w





