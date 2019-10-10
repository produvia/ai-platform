import utils
import cv_model
import data_processing
import argparse
from imports import*


# ap = argparse.ArgumentParser()
# ap.add_argument("-data_path", "--data_path", type = str, help = "path to input data")
# ap.add_argument("-size", "--super_res_crop", type = float, default = 400, help = "final size of image")
# ap.add_argument("-factor", "--upscale_factor", type = float, default = 2, help = "factor to upscale image by (2 or 4)")
# ap.add_argument("-bs", "--bs", type = float, default = 16, help = "batch size")
# ap.add_argument("-epochs", "--epochs", type = float, default = 20, help = "number of epochs")
# ap.add_argument("-load", "--load_model", type = float, default = 1, help = "load model bool")
# ap.add_argument("-device", "--device", type = str, default = 'cpu', help = "device")
# args = vars(ap.parse_args())

keys = 'data_path','super_res_crop','upscale_factor','bs','epochs','load_model','load_model_path','device'
args = {k:v for k,v in zip(keys,sys.argv[1:])}

print()
print('+------------------------------------+')
print('|              PRODUVIA              |')
print('+------------------------------------+')
print()

data_path = Path(args['data_path'])
train_name = 'train'
val_name = 'val'
test_name = 'test'
csv_name = 'dai_super_res.csv'
dp_name = 'DP_super_res.pkl'
train_path = data_path/train_name
csv_path = data_path/csv_name
dp_path = data_path/dp_name

df = pd.DataFrame({'img':[p.name for p in list(train_path.iterdir()) if p.suffix in IMG_EXTENSIONS]})
df.to_csv(csv_path,index=False)
DP = data_processing.DataProcessor(data_path,train_csv=csv_name,tr_name=train_name,setup_data=True)
data_processing.save_obj(dp_path,DP)

DP = data_processing.load_obj(dp_path)
data_dict = DP.data_dict
super_res_crop = int(args['super_res_crop'])
upscale_factor = int(args['upscale_factor'])
bs = int(args['bs'])
epochs = int(args['epochs'])

sets,loaders,sizes = DP.get_data(data_dict=data_dict, bs=bs, dataset=data_processing.dai_super_res_dataset,
                                super_res_crop=super_res_crop,super_res_upscale_factor=upscale_factor)

print_every = sizes[train_name]//3//bs
device = torch.device(args['device'])

models = {
        2:{
            'best_model_folder': 'mlflow_pretrained_models/pretrained_super_res_f2'
            },
        4:{
            'best_model_folder': 'mlflow_pretrained_models/pretrained_super_res_f4'
            }
        }

new_model_path = 'super_res_f{}.pth'.format(args['upscale_factor'])
print("MLflow will save the models in 'mlflow_saved_training_models'")

if args['load_model']:
    if len(args['load_model_path']) > 0:
        load_model_path = args['load_model_path']
    else:
        load_model_path = models[upscale_factor]['best_model_folder']
    net = mlflow.pytorch.load_model(load_model_path,map_location='cpu')
else:
    net = cv_model.SuperRes(model_name = 'sr_model',model_type='super_res',lr=0.05,
                        criterion=nn.L1Loss(),loss_func='perceptual',optimizer_name='sgd',
                        upscale_factor=upscale_factor,
                        growth_rate=64,rdb_number=5,rdb_conv_layers=5,res_blocks=16,
                        device=device,best_model_file=new_model_path)
net.device = device    
net = net.to(device)
lr = net.find_lr(loaders[train_name],plot=False)
net.fit(loaders[train_name],loaders[val_name],epochs=epochs,print_every=print_every)
del net
