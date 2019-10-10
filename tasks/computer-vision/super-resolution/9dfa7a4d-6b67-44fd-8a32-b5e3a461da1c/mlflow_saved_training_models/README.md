### Folder where MLflow will save the best model during training.
The models will have epoch number and time stamp in their name and will be saved whenever the current loss is lesser than the current lowest loss.  
Last saved model will be the best and the others can be ignored/discarded if not needed.
