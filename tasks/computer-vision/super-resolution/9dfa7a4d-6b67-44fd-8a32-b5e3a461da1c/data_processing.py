from imports import*
import utils

class dai_image_dataset(data.Dataset):

    def __init__(self, data_dir, data_df, input_transforms = None, target_transforms = None):
        super(dai_image_dataset, self).__init__()
        self.data_dir = data_dir
        self.data_df = data_df
        self.input_transforms = None
        self.target_transforms = None
        if input_transforms:
            self.input_transforms = transforms.Compose(input_transforms)
        if target_transforms:    
            self.target_transforms = transforms.Compose(target_transforms)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data_df.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')
        target = img.copy()
        if self.input_transforms:
            img = self.input_transforms(img)
        if self.target_transforms:
            target = self.target_transforms(target)
        return img, target

class dai_super_res_dataset(data.Dataset):

    def __init__(self, data_dir, data, input_transform=None, target_transform=None, bicubic_taget_transform=None):
        super(dai_super_res_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.input_transform = transforms.Compose(input_transform)
        self.target_transform = transforms.Compose(target_transform)
        self.bicubic_taget_transform = transforms.Compose(bicubic_taget_transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        # img = load_super_res_img(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        target,bicubic_taget = img.copy(), img.copy()
        if self.input_transform:
            img = self.input_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        if self.bicubic_taget_transform:    
            bicubic_taget = self.bicubic_taget_transform(img)
        return img, target, bicubic_taget  

def csv_from_path(path, img_dest):

    path = Path(path)
    img_dest = Path(img_dest)
    labels_paths = list(path.iterdir())
    tr_images = []
    tr_labels = []
    for l in labels_paths:
        if l.is_dir():
            for i in list(l.iterdir()):
                if i.suffix in IMG_EXTENSIONS:
                    name = i.name
                    label = l.name
                    new_name = '{}_{}'.format(path.name,name)
                    new_path = img_dest/new_name
#                     print(new_path)
                    os.rename(i,new_path)
                    tr_images.append(new_name)
                    tr_labels.append(label)    
            # os.rmdir(l)
    tr_img_label = {'Img':tr_images, 'Label': tr_labels}
    csv = pd.DataFrame(tr_img_label,columns=['Img','Label'])
    csv = csv.sample(frac=1).reset_index(drop=True)
    return csv
    
def add_extension(a,e):
    a = [x+e for x in a]
    return a

def split_df(train_df,test_size = 0.15):
    try:    
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2,stratify = train_df.iloc[:,1])
    except:
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2)
    train_df = train_df.reset_index(drop = True)
    val_df =  val_df.reset_index(drop = True)
    return train_df,val_df    

def save_obj(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class DataProcessor:
    
    def __init__(self, data_path = None, train_csv = None, val_csv = None,test_csv = None,
                 tr_name = 'train', val_name = 'val', test_name = 'test',extension = None, setup_data = True):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        (self.data_path,self.train_csv,self.val_csv,self.test_csv,
         self.tr_name,self.val_name,self.test_name,self.extension) = (data_path,train_csv,val_csv,test_csv,
                                                                      tr_name,val_name,test_name,extension)
        
        self.data_dir = data_path
        if setup_data:
            self.set_up_data()
                
    def set_up_data(self,split_size = 0.15):

        (data_path,train_csv,val_csv,test_csv,tr_name,val_name,test_name) = (self.data_path,self.train_csv,self.val_csv,self.test_csv,
                                                                             self.tr_name,self.val_name,self.test_name)

        # check if paths given and also set paths
        
        if not data_path:
            data_path = os.getcwd() + '/'
        tr_path = os.path.join(data_path,tr_name)
        val_path = os.path.join(data_path,val_name)
        test_path = os.path.join(data_path,test_name)

        if (os.path.exists(os.path.join(data_path,tr_name+'.csv'))) and train_csv is None:
            train_csv = tr_name+'.csv'

        # paths to csv

        if not train_csv:
            # print('no')
            train_csv,val_csv,test_csv = self.data_from_paths_to_csv(data_path,tr_path,val_path,test_path)

        train_csv_path = os.path.join(data_path,train_csv)
        train_df = pd.read_csv(train_csv_path)
        if 'Unnamed: 0' in train_df.columns:
            train_df = train_df.drop('Unnamed: 0', 1)
        if len(train_df.columns) > 2:
            self.obj = True    
        img_names = [str(x) for x in list(train_df.iloc[:,0])]
        if self.extension:
            img_names = add_extension(img_names,self.extension)
        if val_csv:
            val_csv_path = os.path.join(data_path,val_csv)
            val_df = pd.read_csv(val_csv_path)
            val_targets =   list(val_df.iloc[:,1].apply(lambda x: str(x)))
        if test_csv:
            test_csv_path = os.path.join(data_path,test_csv)
            test_df = pd.read_csv(test_csv_path)
            test_targets =   list(test_df.iloc[:,1].apply(lambda x: str(x)))
            
        print('\nSuper Resolution\n')   

        if not val_csv:
            train_df,val_df = split_df(train_df,split_size)
        if not test_csv:    
            val_df,test_df = split_df(val_df,split_size)
        tr_images = [str(x) for x in list(train_df.iloc[:,0])]
        val_images = [str(x) for x in list(val_df.iloc[:,0])]
        test_images = [str(x) for x in list(test_df.iloc[:,0])]
        if self.extension:
            tr_images = add_extension(tr_images,self.extension)
            val_images = add_extension(val_images,self.extension)
            test_images = add_extension(test_images,self.extension)
        train_df.iloc[:,0] = tr_images
        val_df.iloc[:,0] = val_images
        test_df.iloc[:,0] = test_images
        train_df.to_csv(os.path.join(data_path,'{}.csv'.format(self.tr_name)),index=False)
        val_df.to_csv(os.path.join(data_path,'{}.csv'.format(self.val_name)),index=False)
        test_df.to_csv(os.path.join(data_path,'{}.csv'.format(self.test_name)),index=False)
        self.data_dfs = {self.tr_name:train_df, self.val_name:val_df, self.test_name:test_df}
        data_dict = {'data_dfs':self.data_dfs,'data_dir':self.data_dir}
        self.data_dict = data_dict
        return data_dict

    def data_from_paths_to_csv(self,data_path,tr_path,val_path = None,test_path = None):
            
        train_df = csv_from_path(tr_path,tr_path)
        train_df.to_csv(os.path.join(data_path,self.tr_name+'.csv'),index=False)
        ret = (self.tr_name+'.csv',None)
        if val_path is not None:
            val_exists = os.path.exists(val_path)
            if val_exists:
                val_df = csv_from_path(val_path,tr_path)
                val_df.to_csv(os.path.join(data_path,self.val_name+'.csv'),index=False)
                ret = (self.tr_name+'.csv',self.val_name+'.csv')
        if test_path is not None:
            test_exists = os.path.exists(test_path)
            if test_exists:
                test_df = csv_from_path(test_path,tr_path)
                test_df.to_csv(os.path.join(data_path,self.test_name+'.csv'),index=False)
                ret = (self.tr_name+'.csv',self.val_name+'.csv',self.test_name+'.csv')        
        return ret
        
    def get_data(self, data_dict = None, s = (224,224), dataset = dai_super_res_dataset, bs = 32,
                 num_workers = 8, stats_percentage = 0.6, super_res_crop = 256 ,super_res_upscale_factor = 1):
        
        self.image_size = s
        if not data_dict:
            data_dict = self.data_dict
        data_dfs,data_dir = (data_dict['data_dfs'],data_dict['data_dir'])

        super_res_crop = super_res_crop - (super_res_crop % super_res_upscale_factor)
        super_res_transforms = { 
            'input':[
                    transforms.CenterCrop(super_res_crop),
                    transforms.Resize(super_res_crop // super_res_upscale_factor),
                    transforms.ToTensor()
            ],
            'target':[
                    transforms.CenterCrop(super_res_crop),
                    transforms.ToTensor()
            ],
            'bicubic_target':[
                transforms.ToPILImage(),
                transforms.Resize(super_res_crop),
                transforms.ToTensor()
            ]
        }

        image_datasets = {x: dataset(os.path.join(data_dir,self.tr_name),data_dfs[x],
                        super_res_transforms['input'],super_res_transforms['target'],super_res_transforms['bicubic_target'])
                    for x in [self.tr_name, self.val_name, self.test_name]}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                                     shuffle=True, num_workers=num_workers)
                      for x in [self.tr_name, self.val_name, self.test_name]}
        dataset_sizes = {x: len(image_datasets[x]) for x in [self.tr_name, self.val_name, self.test_name]}
        
        self.image_datasets,self.dataloaders,self.dataset_sizes = (image_datasets,dataloaders,dataset_sizes)
        
        return image_datasets,dataloaders,dataset_sizes