import os
import dnnlib.tflib, pickle, torch, collections
import torch.nn as nn
from collections import OrderedDict
from networks.style_gan_net import MappingNet, SynthesisNet

def convert(weights, output_pt_file):
    generator = nn.Sequential(OrderedDict([
        ('g_mapping', MappingNet()),
        ('g_synthesis', SynthesisNet())
    ]))
    if 1:
        # get the weights
        weights_pt = [
            collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k, v in w.trainables.items()]) for w in
            weights]
        torch.save(weights_pt, output_pt_file)
    if 1:
        # convert
        _G, _D, _Gs = torch.load(output_pt_file)
        def key_translate(k, to_print=False):
            k = k.lower().split('/')
            if to_print:
                print(k)
            if k[0] == 'g_synthesis':
                if not k[1].startswith('torgb'):
                    k.insert(1, 'blocks')
                k = '.'.join(k)
                k = (k.replace('const.const', 'const')
                     # early block
                     .replace('const.bias', 'bias')
                     .replace('const.noise.weight', 'epi0.pre_style_op.noise.weight')
                     .replace('const.stylemod.weight', 'epi0.style_mod.linear.module.weight')
                     .replace('const.stylemod.bias', 'epi0.style_mod.linear.bias')
                     #.replace('const.stylemod', 'epi0.style_mod.linear')
                     .replace('conv.weight', 'conv.module.weight')
                     .replace('conv.noise.weight', 'epi1.pre_style_op.noise.weight')
                     .replace('conv.stylemod.weight', 'epi1.style_mod.linear.module.weight')
                     .replace('conv.stylemod.bias', 'epi1.style_mod.linear.bias')
                     # later blocks
                     .replace('conv0_up.weight', 'conv0_up.conv.module.weight')
                     .replace('conv0_up.bias', 'conv0_up.conv.bias')
                     .replace('conv0_up.noise.weight', 'epi0.pre_style_op.noise.weight')
                     .replace('conv0_up.stylemod.weight', 'epi0.style_mod.linear.module.weight')
                     .replace('conv0_up.stylemod.bias', 'epi0.style_mod.linear.bias')

                     .replace('conv1.weight', 'conv1.module.weight')
                     #.replace('conv1.bias', 'conv1.module.bias')
                     .replace('conv1.noise.weight', 'epi1.pre_style_op.noise.weight')
                     .replace('conv1.stylemod.weight', 'epi1.style_mod.linear.module.weight')
                     .replace('conv1.stylemod.bias', 'epi1.style_mod.linear.bias')
                     #.replace('torgb_lod0', 'torgb')
                     .replace('torgb_lod0.weight', 'torgb.module.weight')
                     .replace('torgb_lod0.bias', 'torgb.bias')
                      )
            else:
                # mapping net
                if k[0] == 'g_mapping' and k[2] == 'weight':
                    k.insert(2, 'module')
                k = '.'.join(k)
            return k

        def weight_translate(k, w):
            k = key_translate(k)
            if k.endswith('.weight'):
                if w.dim() == 2:
                    w = w.t()
                elif w.dim() == 1:
                    pass
                else:
                    assert w.dim() == 4
                    w = w.permute(3, 2, 0, 1)
            return w

        # we delete the useless torgb filters
        param_dict = {key_translate(k, True): weight_translate(k, v) for k, v in _Gs.items() if
                      'torgb_lod' not in key_translate(k)}
        for k, v in param_dict.items():
            print('pd ', k, v.shape)
        if 1:
            sd_shapes = {k: v.shape for k, v in generator.state_dict().items()}
            param_shapes = {k: v.shape for k, v in param_dict.items()}

            for k in list(sd_shapes) + list(param_shapes):
                pds = param_shapes.get(k)
                sds = sd_shapes.get(k)
                if pds is None:
                    print("sd only", k, sds)
                elif sds is None:
                    print("pd only", k, pds)
                elif sds != pds:
                    print("mismatch!", k, pds, sds)

        generator.load_state_dict(param_dict, strict=False)  # needed for the blur kernels
        torch.save(generator.state_dict(), output_pt_file)



if __name__ == '__main__':
    to_convert = True
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl

    pretrained_dir = 'pretrained'
    output_pt_file = os.path.join(pretrained_dir, 'karras2019stylegan-ffhq-1024x1024.pt')
    if to_convert:
        # init tf
        print('start conversion')
        dnnlib.tflib.init_tf()
        with dnnlib.util.open_url(url, cache_dir=pretrained_dir) as f:
            weights = pickle.load(f)
            convert(weights, output_pt_file)
        print('finished conversion')
    generator = nn.Sequential(OrderedDict([
        ('g_mapping', MappingNet(resolution=1024)),
        ('g_synthesis', SynthesisNet(resolution=1024))
    ]))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    generator.eval()
    generator.to(device)
    torch.manual_seed(7)
    latents = torch.randn(1, 512, device=device)
    with torch.no_grad():
        imgs = generator(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0

    imgs = imgs.cpu()

