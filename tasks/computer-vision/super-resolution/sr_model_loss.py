from imports import*


def flatten(x): return x.view(x.size(0), -1)

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()        

class FeatureLoss(nn.Module):
    def __init__(self, max_layer_id, layer_wgts, device = 'cpu'):
        super().__init__()
        m_vgg = vgg16_bn(True)
        blocks = [i-1 for i,o in enumerate(list(m_vgg.features.children()))
                    if isinstance(o,nn.MaxPool2d)][:max_layer_id]
        vgg_layers = list(m_vgg.features.children())[:13]
        m_vgg = nn.Sequential(*vgg_layers).to(device).eval()
        self.m,self.wgts = m_vgg,layer_wgts
        self.sfs = [SaveFeatures(m_vgg[i]) for i in blocks]

    def forward(self, input, target, sum_layers=True):
        self.m(target)
        res = [F.l1_loss(input,target)]
        # mse_loss = F.mse_loss(input,target)
        # psnr = (34.0-(10 * math.log10(1 / mse_loss)))*0.5
        # res = [psnr]
        targ_feat = [(o.features.data.clone()) for o in self.sfs]
        self.m(input)
        res += [(F.l1_loss(flatten(inp.features),flatten(targ))*wgt)*0.01
               for inp,targ,wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers: res = sum(res)
        return res
    
    def close(self):
        for o in self.sfs: o.remove()
            
# learn.crit = FeatureLoss(m_vgg, blocks[:2], [0.26,0.74])