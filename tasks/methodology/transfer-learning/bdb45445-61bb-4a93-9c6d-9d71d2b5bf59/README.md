AI-Platform UUID: `bdb45445-61bb-4a93-9c6d-9d71d2b5bf59`

# Transfer Learning

## Goal

The goal of this task is: Transfer knowledge from a well-labeled domain to a similar but different domain with limited or no labels. 

As an example, we show how to transfer knowledge in an image classification task, where we reuse the weights learnt on one dataset (ImageNet) to classify a different dataset (CIFAR-10). 

## Base CNN network

We use the well-known VGG16 network trained on the ImageNet dataset. The weights are available in Keras via the keras.applications module. 

The new model is built using the convolutional layers of the VGG16 model and adding two new dense layers (to be able to output decisions relevant to the CIFAR-10 classification problem)

To avoid degradation of the convolutional weights when training the new dense layers, the convolutional layers are frozen (some can be retrained via the retrained_conv_layers input parameter) so the training focuses on modifying the weights of the dense layers.


## Output

Running the model transfers the knowledge in the convolutional layers from VGG16 to a new model, trains it towards the new classification task (CIFAR-10), evaluates its performance and stores the new weights. The weights are stored in the model subfolder (model/model.h5)

## How to run

Using mlflow:

`mlflow run . -P samples_per_class=300 -P retrained_conv_layers=0 -P use_early_stop=1 -P max_epochs=20`

By default, samples_per class is 300, retrained_conv_layers is 0, use_early_stop is 1 (true) and max_epochs is 20. samples_per_class is used to speed up training by only selecting a subset of samples per class to train on. retrained_conv_layers is used to determined how many of the final convolutional layers transferred should be retrained. use_early_stop decides whether training should use early stop (to avoid overfitting). max_epochs limits the maximum number of training epochs.



## References

- [ImageNet](http://www.image-net.org/) ImageNet dataset.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) CIFAR-10 dataset
- [VGG16](https://keras.io/applications/#vgg16) VGG16 in keras, reference: Simonyan, K. and Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition.  	arXiv:1409.1556 (cs.CV)
