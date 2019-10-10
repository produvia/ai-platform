from imports import*
from utils import *
from model import *
from rdn import RDN, RDN_DN
import sr_model
import sr_model_loss

class SuperRes(Network):
    def __init__(self,
                 model_name = 'sr_model',
                 model_type = 'super_res',
                 lr = 0.02,
                 criterion = nn.L1Loss(),
                 perceptual_criterion = nn.L1Loss(),
                 loss_func = 'perecptual',
                 optimizer_name = 'sgd',
                 upscale_factor = 3,
                 growth_rate = 64,
                 rdb_number = 16,
                 rdb_conv_layers = 8,
                 res_blocks = 10,
                 device = None,
                 best_validation_loss = None,
                 best_model_file = 'best_super_res_sgd.pth'
                 ):

        super().__init__(device=device)
        self.set_model(model_name=model_name,upscale_factor=upscale_factor,growth_rate=growth_rate,
                       rdb_number=rdb_number,rdb_conv_layers=rdb_conv_layers,res_blocks=res_blocks,device=device)
        self.set_model_params(criterion = criterion,optimizer_name = optimizer_name,lr = lr,model_name = model_name,model_type = model_type,
                              best_validation_loss = best_validation_loss,best_model_file = best_model_file)
        if loss_func.lower() == 'perceptual':
            self.feature_loss = sr_model_loss.FeatureLoss(2,[0.26,0.74],device = device)
        self.best_psnr = None
        self.loss_func = loss_func
        self.model = self.model.to(device)

    def set_model(self,model_name,upscale_factor,growth_rate=64,
                  rdb_number=16,rdb_conv_layers=8,res_blocks=10,device='cpu'):
        print('Setting up Super Resolution model: ',end='')
        if model_name.lower() == 'rdn':
            print('Using RDN for super res.')
            self.model = RDN(channel=3,growth_rate=growth_rate,rdb_number=rdb_number,
                            rdb_conv_layers=rdb_conv_layers,upscale_factor=upscale_factor).to(device)
        elif model_name.lower() == 'sr_model':
            print('Using SrResnet for super res.')
            self.model = sr_model.SrResnet(scale=upscale_factor,res_blocks=res_blocks).to(device)
            
    def forward(self,x):
        return self.model(x)

    def compute_loss(self,criterion,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        if self.loss_func.lower() == 'crit':
            basic_loss = criterion(outputs, labels)
            ret['overall_loss'] = basic_loss
            return basic_loss,ret
        elif self.loss_func.lower() == 'perceptual':
            overall_loss = self.feature_loss(outputs,labels)
            ret['overall_loss'] = overall_loss
            return overall_loss,ret
    
    def evaluate(self,dataloader):
        
        running_loss = 0.
        running_psnr = 0.
        rmse_ = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in dataloader:
                img, hr_target, hr_bicubic = data_batch[0],data_batch[1],data_batch[2]
                img = img.to(self.device)
                hr_target = hr_target.to(self.device)
                hr_super_res = self.forward(img)
                loss,loss_dict = self.compute_loss(self.criterion,hr_super_res,hr_target)
                torchvision.utils.save_image([hr_target.cpu()[0],hr_bicubic[0],hr_super_res.cpu()[0]],filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret