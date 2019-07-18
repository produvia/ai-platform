import os
import dnnlib.tflib, pickle, torch, collections
from matplotlib import pyplot as plt
import torchvision
from networks.style_gan_net import Generator, BasicDiscriminator

def convert(weights, generator, g_out_file, discriminator, d_out_file):
    # get the weights
    weights_pt = [
        collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k, v in w.trainables.items()]) for w in
        weights]
    torch.save(weights_pt, g_out_file)
    # convert
    _G, _D, _Gs = torch.load(g_out_file)
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
        elif k[0] == 'g_mapping':
            # mapping net
            if k[2] == 'weight':
                k.insert(2, 'module')
            k = '.'.join(k)
        # discriminator
        else:
            k = '.'.join(k)
            k = (k.replace('fromrgb_lod0.weight', 'fromrgb.module.weight')
                  .replace('fromrgb_lod0.bias', 'fromrgb.bias')
                  .replace('conv0.weight', 'conv0.module.weight')
                  .replace('conv1_down.weight', 'conv1_down.conv.module.weight')
                  .replace('conv1_down.bias', 'conv1_down.conv.bias')
                  .replace('conv.weight', 'conv.module.weight')
                  .replace('dense0.weight', 'dense0.module.weight')
                  .replace('dense1.weight', 'dense1.module.weight')
                 )
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

    # TODO: Note that for training, our structure is fixed. Needs to support growing.
    def translate_checkpoint_with_defined(checkpoint, defined):
        for k, v in checkpoint.items():
            print('original key in checkpoint: ', k)

        checkpoint_pd = {key_translate(k, True): weight_translate(k, v) for k, v in checkpoint.items()}

        # we delete the useless torgb_(1-10) and fromrgb(0-9) conv filters which are used for growing training
        checkpoint_pd = {k: v for k, v in checkpoint_pd.items() if not ('torgb_lod' in k or 'fromrgb_lod' in k)}
        for k, v in checkpoint_pd.items():
            print('checkpoint parameter ', k, v.shape)
        if 1:
            defined_shapes = {k: v.shape for k, v in defined.state_dict().items()}
            param_shapes = {k: v.shape for k, v in checkpoint_pd.items()}

            for k in list(defined_shapes) + list(param_shapes):
                pds = param_shapes.get(k)
                dss = defined_shapes.get(k)
                if pds is None:
                    print("ours only", k, dss)
                elif dss is None:
                    print("theirs only", k, pds)
                elif dss != pds:
                    print("mismatch!", k, pds, dss)
        return checkpoint_pd

    # translate generator
    if generator is not None:
        g_checkpoint_pd = translate_checkpoint_with_defined(checkpoint=_Gs, defined=generator)
        # strict needs to be False for the blur filters
        generator.load_state_dict(g_checkpoint_pd, strict=False)
        torch.save(generator.state_dict(), g_out_file)

    if discriminator is not None:
        d_checkpoint_pd = translate_checkpoint_with_defined(checkpoint=_D, defined=discriminator)
        discriminator.load_state_dict(d_checkpoint_pd, strict=False)
        torch.save(discriminator.state_dict(), d_out_file)



if __name__ == '__main__':
    to_convert = False
    # changes this accordingly
    url_choice = 'bedrooms'
    checkpoint_prefix = 'kerras2019stylegan'
    url = {'cats': ('https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ',
                    256,
                    '1c6bbbad79102cf05f29ec0363071cf3'),
           'bedrooms': ('https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF',
                        256,
                        '258371067819a08c899eeb7d1d2c8c19'
                        ),
           'ffhq': ('https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
                    1024,
                    '263e666dc20e26dcbfa514733c1d1f81'
                    )
           }[url_choice]

    pretrained_dir = 'pretrained'
    g_out_file = os.path.join(pretrained_dir,
                              'karras2019stylegan-{}-{}x{}.generator.pt'.format(url_choice, url[1], url[1]))
    d_out_file = os.path.join(pretrained_dir,
                              'karras2019stylegan-{}-{}x{}.discriminator.pt'.format(url_choice, url[1], url[1]))

    generator = Generator(resolution=url[1])
    print(generator)
    pass
    discriminator = BasicDiscriminator(resolution=url[1])

    if to_convert:
        # init tf
        print('start conversion')
        dnnlib.tflib.init_tf()
        try:
            with dnnlib.util.open_url(url[0], cache_dir=pretrained_dir) as f:
                weights = pickle.load(f)

        except:
            weights = pickle.load(open(os.path.join(pretrained_dir,
                                                    'karras2019stylegan-{}-{}x{}.pkl'.format(url_choice, url[1], url[1]))))
        convert(weights,
                generator=generator,
                g_out_file=g_out_file,
                discriminator=discriminator,
                d_out_file=d_out_file)

        print('finished conversion')

    generator.load_state_dict(torch.load(g_out_file))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    generator.eval()
    generator.to(device)
    torch.manual_seed(77)
    nrow=2
    ncol=2

    latents = torch.randn(nrow * ncol, 512, device=device)
    with torch.no_grad():
        imgs = generator(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0

    output_imgs = imgs
    imgs = torchvision.utils.make_grid(output_imgs.cpu(), nrow=nrow)

    plt.figure(figsize=(30, 12))
    plt.imshow(imgs.permute(1, 2, 0).detach().numpy())
    plt.show()

    discriminator.load_state_dict(torch.load(d_out_file))

    discriminator.eval()
    discriminator.to(device)
    result = discriminator(output_imgs).cpu().detach().numpy()
    print(result)

