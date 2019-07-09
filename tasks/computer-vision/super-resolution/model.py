from imports import *
from utils import *

class Network(nn.Module):
    def __init__(self,device=None):
        super().__init__()
        self.parallel = False
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(self.device)

    def forward(self,x):
        pass
    def compute_loss(self,criterion,outputs,labels):
        return [criterion(outputs,labels)]

    def fit(self,trainloader,validloader,epochs=2,print_every=10,validate_every=1,save_best_every=1):

        optim_path = Path(self.best_model_file)
        optim_path = optim_path.stem + '_optim' + optim_path.suffix
        with mlflow.start_run() as run:
            for epoch in range(epochs):
                self.model = self.model.to(self.device)
                mlflow.log_param('epochs',epochs)
                mlflow.log_param('lr',self.optimizer.param_groups[0]['lr'])
                mlflow.log_param('bs',trainloader.batch_size)
                print('Epoch:{:3d}/{}\n'.format(epoch+1,epochs))
                epoch_train_loss =  self.train_((epoch,epochs),trainloader,self.criterion,
                                                self.optimizer,print_every)
                        
                if  validate_every and (epoch % validate_every == 0):
                    t2 = time.time()
                    eval_dict = self.evaluate(validloader)
                    epoch_validation_loss = eval_dict['final_loss']
                    epoch_psnr = eval_dict['psnr']
                    mlflow.log_metric('Train Loss',epoch_train_loss)
                    mlflow.log_metric('Valdiation Loss',epoch_validation_loss)
                    mlflow.log_metric('Valdiation PSNR',epoch_psnr)
                    
                    time_elapsed = time.time() - t2
                    if time_elapsed > 60:
                        time_elapsed /= 60.
                        measure = 'min'
                    else:
                        measure = 'sec'    
                    print('\n'+'/'*36+'\n'
                            f"{time.asctime().split()[-2]}\n"
                            f"Epoch {epoch+1}/{epochs}\n"    
                            f"Validation time: {time_elapsed:.6f} {measure}\n"    
                            f"Epoch validation psnr: {epoch_psnr:.6f}\n"
                            f"Epoch training loss: {epoch_train_loss:.6f}\n"                        
                            f"Epoch validation loss: {epoch_validation_loss:.6f}"
                        )

                    print('\\'*36+'\n')
                    # if self.best_psnr == None or (epoch_psnr >= self.best_psnr):
                    #     print('\n**********Updating best validation psnr**********\n')
                    #     if self.best_psnr is not None:
                    #         print('Previous best: {:.7f}'.format(self.best_psnr))
                    #     print('New best psnr = {:.7f}\n'.format(epoch_psnr))
                    #     print('*'*49+'\n')
                    #     self.best_psnr = epoch_psnr
                    if self.best_validation_loss == None or (epoch_validation_loss <= self.best_validation_loss):
                        print('\n**********Updating best validation loss**********\n')
                        if self.best_validation_loss is not None:
                            print('Previous best: {:.7f}'.format(self.best_validation_loss))
                        print('New best loss = {:.7f}\n'.format(epoch_validation_loss))
                        print('*'*49+'\n')
                        self.best_validation_loss = epoch_validation_loss
                        mlflow.log_metric('Best Loss',self.best_validation_loss)
                        optim_path = Path(self.best_model_file)
                        optim_path = optim_path.stem + '_optim' + optim_path.suffix
                        torch.save(self.model.state_dict(),self.best_model_file)
                        torch.save(self.optimizer.state_dict(),optim_path)     
                        mlflow.pytorch.log_model(self,'mlflow_logged_models')#/Path(self.best_model_file).stem)
                        curr_time = str(datetime.now())
                        curr_time = '_'+curr_time.split()[1].split('.')[0]
                        mlflow_save_path = Path('mlflow_saved_training_models')/(Path(self.best_model_file).stem+'_{}'.format(str(epoch)+curr_time))
                        mlflow.pytorch.save_model(self,mlflow_save_path)
                        
                    self.train()
        torch.cuda.empty_cache()
        print('\nLoading best model\n')
        self.model.load_state_dict(torch.load(self.best_model_file))
        self.optimizer.load_state_dict(torch.load(optim_path))
        os.remove(self.best_model_file)
        os.remove(optim_path)

    def train_(self,e,trainloader,criterion,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,labels = data_batch[0],data_batch[1]
            batches += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(criterion,outputs,labels)[0]    
            loss.backward()
            loss = loss.item()
            optimizer.step()
            running_loss += loss
            if batches % print_every == 0:
                elapsed = time.time()-t1
                if elapsed > 60:
                    elapsed /= 60.
                    measure = 'min'
                else:
                    measure = 'sec'
                batch_time = time.time()-t0
                if batch_time > 60:
                    batch_time /= 60.
                    measure2 = 'min'
                else:
                    measure2 = 'sec'    
                print('+----------------------------------------------------------------------+\n'
                        f"{time.asctime().split()[-2]}\n"
                        f"Time elapsed: {elapsed:.3f} {measure}\n"
                        f"Epoch:{epoch+1}/{epochs}\n"
                        f"Batch: {batches+1}/{len(trainloader)}\n"
                        f"Batch training time: {batch_time:.3f} {measure2}\n"
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader) 

    def predict(self,inputs):
        self.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
        return outputs

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optimizer = self.optimizer
        criterion = self.criterion
        num = min(1,len(trn_loader)-1)
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data in trn_loader:
            batch_num += 1
            inputs,labels = data[0],data[1]
            inputs = inputs.to(self.device)           
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(criterion,outputs,labels)[0]
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr
            
    def plot_find_lr(self):    
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.log_lrs,self.find_lr_losses)
        plt.show()
                
    def set_criterion(self, criterion):
        if criterion:
            self.criterion = criterion
        
    def set_optimizer(self,params,optimizer_name='adam',lr=0.003):
        if optimizer_name:
            if optimizer_name.lower() == 'adam':
                print('Setting optimizer: Adam')
                self.optimizer = optim.Adam(params,lr=lr)
                self.optimizer_name = optimizer_name
            elif optimizer_name.lower() == 'sgd':
                print('Setting optimizer: SGD')
                self.optimizer = optim.SGD(params,lr=lr)
            elif optimizer_name.lower() == 'adadelta':
                print('Setting optimizer: AdaDelta')
                self.optimizer = optim.Adadelta(params)       
            
    def set_model_params(self,
                         criterion = nn.CrossEntropyLoss(),
                         optimizer_name = 'sgd',
                         lr = 0.01,
                         model_name = 'resnet50',
                         model_type = 'classifier',
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth'):        
        self.set_criterion(criterion)
        self.optimizer_name = optimizer_name
        self.set_optimizer(self.parameters(),optimizer_name,lr=lr)
        self.lr = lr
        self.model_name =  model_name
        self.model_type = model_type
        self.best_validation_loss = best_validation_loss
        self.best_model_file = best_model_file
    
    def get_model_params(self):
        params = {}
        params['device'] = self.device
        params['model_type'] = self.model_type
        params['model_name'] = self.model_name
        params['optimizer_name'] = self.optimizer_name
        params['criterion'] = self.criterion
        params['lr'] = self.lr
        params['best_validation_loss'] = self.best_validation_loss
        params['best_model_file'] = self.best_model_file
        return params
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
