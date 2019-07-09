import cv_model
import utils
import argparse
from imports import*

keys = 'input_path','upscale_factor','output_path','load_model_path','device'
args = {k:v for k,v in zip(keys,sys.argv[1:])}

print()
print('+------------------------------------+')
print('|              PRODUVIA              |')
print('+------------------------------------+')
print()

device = torch.device(args['device'])
upscale_factor = int(args['upscale_factor'])
input_path = args['input_path']

def process_batch(batch, show = False):
    output = np.clip(utils.tensor_to_img(batch)[0],0.,1.)
    plt.imsave(args['output_path'],output)
    if show:
        utils.plt_show(output)
    return output    

batch = utils.get_test_input(paths=[input_path])
img = batch[0]
img = utils.tensor_to_img(img)
shutil.copy2(input_path,'original'+Path(input_path).suffix)
batch = batch.to(device)

models = {
        2:{
            'best_model_folder': 'mlflow_pretrained_models/pretrained_super_res_f2'
            },
        4:{
            'best_model_folder': 'mlflow_pretrained_models/pretrained_super_res_f4'
            }
        }

if len(args['load_model_path']) > 0:
    load_model_path = args['load_model_path']
else:
    load_model_path = models[upscale_factor]['best_model_folder']

net = mlflow.pytorch.load_model(load_model_path,map_location='cpu')
net.device = device
net = net.to(device)
net.eval()
sr = net(batch)
sr = sr.detach().cpu().numpy()
img = img.numpy()
resized = cv2.resize(img,(img.shape[1]*upscale_factor,img.shape[0]*upscale_factor))
output = process_batch(sr)
utils.plot_in_row(imgs=[img,resized,output],titles=['Low Resolution Image','{}x Resized Image'.format(str(upscale_factor)),
                        '{}x Super Resolution Image'.format(str(upscale_factor))],fig_path='sr_fig.png')
del net
del batch
del sr
