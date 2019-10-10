from imports import*

def display_img_actual_size(im_data,title = ''):
    dpi = 80
    height, width, depth = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.title(title,fontdict={'fontsize':25})
    plt.show()

def plt_show(im):
    plt.imshow(im)
    plt.show()

def load_and_show(path):
    img = plt.imread(path)
    plt_show(img)    

def denorm_img_general(inp):
    inp = inp.numpy()
    inp = inp.transpose((1, 2, 0))
    mean = np.mean(inp)
    std = np.std(inp)
    inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    return inp 

def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)    

def plot_in_row(imgs,figsize = (20,20),rows = None,columns = None,titles = [],fig_path = 'fig.png'):
    fig=plt.figure(figsize=figsize)
    if len(titles) == 0:
        titles = ['image_{}'.format(i) for i in range(len(imgs))]
    if not rows:
        rows = 1
    if not columns:    
        columns = len(imgs)
    for i in range(1, columns*rows +1):
        img = imgs[i-1]
        fig.add_subplot(rows, columns, i, title = titles[i-1])
        plt.imshow(img)
    fig.savefig(fig_path)    
    plt.show()

def tensor_to_img(t):
    if len(t.shape) > 3:
        return [np.transpose(t_,(1,2,0)) for t_ in t]
    return np.transpose(t,(1,2,0))

def get_test_input(paths = [],imgs = [], size = None, size_factor = None, show = False, norm = False, bgr_to_rgb = False):
    if len(paths) > 0:
        bgr_to_rgb = True
        imgs = []
        for p in paths:
            imgs.append(cv2.imread(str(p)))
    for i,img in enumerate(imgs):
        if size:
            img = cv2.resize(img, size)
        if size_factor:
            img = cv2.resize(img, (img.shape[1]//size_factor,img.shape[0]//size_factor))        
        if bgr_to_rgb:
            img = bgr2rgb(img)
        if show:
            plt_show(img)    
        img_ =  img.transpose((2,0,1)) # H X W C -> C X H X W
        if norm:
            img_ = (img_ - np.mean(img_))/np.std(img_)
        imgs[i] = img_/255.
    return torch.from_numpy(np.asarray(imgs)).float()

def to_batch(paths = [],imgs = [], size = None):
    if len(paths) > 0:
        imgs = []
        for p in paths:
            imgs.append(cv2.imread(p))
    for i,img in enumerate(imgs):
        if size:
            img = cv2.resize(img, size)
        img =  img.transpose((2,0,1))
        imgs[i] = img
    return torch.from_numpy(np.asarray(imgs)).float()    

def get_optim(optimizer_name,params,lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(params=params,lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params=params,lr=lr)
    elif optimizer_name.lower() == 'adadelta':
        return optim.Adadelta(params=params)

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

class Printer(nn.Module):
    def forward(self,x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def rmse(inputs,targets):
    return torch.sqrt(torch.mean((inputs - targets) ** 2))

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_psnr(inputs,targets):
    mse_loss = F.mse_loss(inputs,targets)
    return 10 * math.log10(1 / mse_loss)