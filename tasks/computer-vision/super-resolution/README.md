# Super Resolution using convolutional neural networks.

Increase image resolution x2 or x4 while maintaining or even improving image quality.

### Important Note:

Make sure to first download the models from the provided links:

Super Resolution x2: https://drive.google.com/drive/folders/1Su9LjUUOarwRasRsnJvPBnJN34g0aawr?usp=sharing Super Resolution x4: https://drive.google.com/drive/folders/17jlTPybEkomlTc7kh75Jiy5fv886Uqcd?usp=sharing

Download the whole folders 'pretrained_super_res_f4' and 'pretrained_super_res_f2' and paste them in 'mlflow_pretrained_models' without changing their names.

## Usage

### Train:

List of parameters: 

  data_path: Full path to the directory of training images. Make sure to keep all images in a folder named 'train' before running script. e.g. if images are in '/training_data/train/', data_path = '/training_data/'.  
  
  size: Size of final image after super resolution. (Default = 400)  
  
  factor: Upscale factor (2 or 4). e.g. If size  = 1024 and upscale factor = 4, model will learn to turn image from (256,256) to (1024,1024) (Default = 2)  
  
  bs: Batch size. (Default = 32)  
  
  epochs: Number of epochs to train. (Default = 20)  
  
  load: Boolean whether or not to load a pretrained model. (Default = True)  
  
  load_path: Directory where MLflow has saved a previous model for loading later. MLflow will save the models you train in 'mlflow_saved_traing_models'. (Default = '', so it will load from 'mlflow_pretrained_models/pretrained_super_res_f2' by default.)   
  
  device: Device to run the code on ('cpu', 'cuda', 'cuda:0' etc.) (Default = 'cpu')
  
#### Command:    

mlflow run . -e train -P data_path='/training_data/' -P size=512 -P factor=2 -P bs=10 -P epochs=20 -P device='cpu'

### Super Resolution

List of parameters: 

  input_path: Path of input low resolution.  
  
  factor: Upscale factor (2 or 4). (Default = 2)  
  
  output_path: Path of output super resolution image. (Default = 'output.png')  
        
  load_path: Directory where MLflow has saved a previous model for laoding later. MLflow will save the models you train in 'mlflow_saved_traing_models'. (Default = '', so it will load from 'mlflow_pretrained_models/pretrained_super_res_f2' by default.)  
  
  device: Device to run the code on ('cpu', 'cuda', 'cuda:0' etc.) (Default = 'cpu')  
  
#### Command:    

mlflow run . -e super_res -P input_path='example_input.png' -P factor=2 -P device='cpu'

### Important Note:
  
The model itself is quite big and super resolution is a memory heavy task.  
In my experiments while training on an 8GB GPU, I was unable to go beyond a batch size of 8-10 with parameter size=300 without my GPU going out of memory.  
Factor 2 will generally require a bit more memory than factor 4, so in my case if factor 2 can have bs=8, factor 4 could have bs=10.  

For super resolution, low resolution images smaller than 720 work just fine.
