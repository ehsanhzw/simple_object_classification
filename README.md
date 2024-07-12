# Object Classification using Python Tensorflow
In this mini-project I intend to test some neural network models to classify two different objects such as images of cats or dogs. My first priority is to maximize the validation accuracy of models of this project.  

## Models
The models are based on commonly used networks and modified a bit to satisfy our desired accuracy. Currently the models have a maximum validation accuracy of 81% for cat/dog classification problem.

Convolutional neural networks I used for this problem can overally be seperated into two groups, CNN's and U-Net based models which are also a kind of CNN's. For the CNN's, **2D convolution** layers are immediately followed by a **2D maxpooling** layers which simply thorows away unused data and reduces the width and height of the convolution layer and redundancies.

<p align="center">
  <img src="Doc/cnn.jpg" alt="CNN" />
  <em>CNN Model</em>
</p>

In U-Net based models, two convolutional layers are cascaded first, then maxpooling is applied so they get a shape of U as they get down tothe output. I will gradually add and test more models in the future.

<p align="center">
  <img src="Doc/unet_1.jpg" alt="U-Net" />
  <em>U-Net Model</em>
</p>

Actually, U-Net model is well known for its U shape and upsampling path which simply includes the deconvolutions concatenated with outputs of conv2D filters. As we don't intend to do the segmentation task in this project, this parts of U-Net architecture are not seen in our models and only the encoding path is included. This fact is also applied to moder U-Net architecture such as U-Net++ and so on. 

## How to Use
trainer.py is used to train the model with the cifar-10 datasets. It uses chosen_classes list to train the network based on the two classes we mention in this list. If you want to change the network you want to train, just change the model_name in first lines of code. Also you may want to change epochs and files names. To train the network for all 10 outputs of the datasets, simply remove the selector lines.

In order to get a sense of model performance, jump into the validation.py script and change the model_name according to the model you want to test. This file uses the images in the project which are manually gathered from the internet to test the network. Make sure the paths are correct in the first lines of this code. Also, if you trained the network for other objects such as planes and cars, change the images_path variable and make sure the images can be found in that directory and don't include anything except images in that folder. 

<p align="center">
  <img src="models/unet/unet_CatDog_validation.png" alt="resulting test in validation.py with u-net model" />
  <em>U-Net Model Test Results for Cat/Dog</em>
</p>

<p align="center">
  <img src="models/quickcnn/quickcnn_AirplaneAutomobile_validation.png" alt="resulting test in validation.py with quick cnn-net model" />
  <em>Quick CNN Model Test Results for Plane/Car</em>
</p>

Models are implemented in nn_models file. If you want to add models, write a function of your model in nn_models.py, then, add it to match case statement of trainer.py, add it to the model names in first lines of the code commented for easier access.

### Dependencies
- numpy
- matplotlib
- tensorflow (keras)
- opencv
- pydot
- graphvis for your os

## Outputs
Trainer gives a summary of network while running. After the model gets trained, it plots the error and accuracy of train and test data and saves it in model directory. Validation also saves the manual test output in model directory.

Overally, it can be seen that model isn't trained soo well in Cat/Dog test, but is very good in other test such as Car/Plane test. It can be predicted that e.g. if we test the network with Deer/Horse, it will not pass an accuracy of 70-80%, but will perfectly work in e.g. ship/Truck and Bird/Frog case.  

<p align="center">
  <img src="models/unet/unet_CatDog_performance.png" alt="cnn" />
  <em>U-Net Accuracy/Loss Per Epoch for Cat/Dog</em>
</p>

<p align="center">
  <img src="models/quickcnn/quickcnn_AirplaneAutomobile_performance.png" alt="cnn" />
  <em>Quick CNN Model Accuracy/Loss Per Epoch for Plane/Car</em>
</p>

## Probable Ways to Improve
Cats and dogs are visually more similar to each other than, for example, planes and cars. This similarity makes the design of optimum neural network very complex. Considering tons of breeds out there with many appearance distinctions, we naturally guess that the model must be a bit more complicated than a typical cnn. But, going too hard with size of cnn, overfitting might occur and if we look at the problem very simplisticly, it leads to underfitting of network.

By far, the accuracy of model for cats and dogs classification, increased in microscopic scale with this procedure of testing models. I have a limited ideas at the moment that may fix the issue (Or not).

### Augmentation Techniques
Applying various data augmentation techniques (like rotation, flipping, zooming, etc.) to increase the diversity of training data. This may help the network generalize better. But if we are not careful with the augmentation process, it leads to greater noise and more problems.
### Network Architecture
Tens of models are tested by now and the improvements are very tiny and ignorable. I already demonstrated that the models work perfectly with other datasets. However, I include common architectural problems that the networks may have been suffering from. For instance, the chance of overfitting is high, but as I shrink the network, after the threshold is passed, the network gets quickly underfit. So finding this threshold is essential if achievable.
- Network Complexity: It might be that the architecture is too simple or too complex for the task. I should check that the network has the right balance of depth and width in future tests.
- Overfitting/Underfitting: It might be that network is overfitting or underfitting. Overfitting can be addressed with regularization techniques like dropout, L1/L2 regularization, etc. 
- Pre-trained Models: Using a pre-trained model like VGG, ResNet, or EfficientNet and fine-tuning it on cat/dog dataset is one solution. I skip it for now because I am not familiar with those networks and the effort might be useless.
### Training Process
- Learning Rate: Experimenting with different learning rates. Worth a shot, in the future tests. Learning rate that's too high or too low can lead to low accuracy.
- Loss Function: Checking that an appropriate loss function is used for the classification task or not.
- Optimizer: Different optimizers can lead to different results. I used Adam, but other options to check are RMSprop, SGD with momentum, or some others.
- Early stopping: Meaning that we stop as soon as loss started to grow steadily. This is not included yet, but from the performance plots, we can estimate the number of epochs to achieve the best accuracy. I will include this in the future to make the training process more active.
### Evaluation and Validation
- Cross-Validation: Using cross-validation to ensure that results are consistent across different splits of the dataset.
### Fixing Probable Overfit
As the accuracy and loss plots are shown above, it seems that in some cases overfit occured. However, the loss is not that much decreased after modifications on network and regularization methods. I spent a while now on finding methods to fix this. The methods were regularizatiion, simplification of network, using other optimizers, dropout, and early stopping, which of course I tried them all and the improvements were 2-3% and the best result was one of the tests in models directory with 82.6% val_acc. However the ultimate fix of the problem that seems to be the right approach is train data samples augmentation and using another dataset besides this one. All the good results that I found after a bit of search used another dataset from archives on the web.
