# Machine Learning 
## 01_Model on titanic dataset
I have tried to build a model using RandomForestRegressor based on titanic dataset which gives a prediction on survival. RandomForestRegressor is a machine learning algorithm that works by creating a team of decision makers called "decision trees." 


## 02_RandomForestClassifier
First I have trained a Random Forest Classifier on the dataset of MNIST_784 . Then applied PCA to reduce the dataset's dimensionality, with an explained variance ratio of 95%. And again trained Random Forest Classifier on the reduced dataset.


## 03_RandomizedSearchCV
I have tried to find the hyperparameters and the corresponding best model using RandomizedSearchCV. Where PCA and Random Forest Classifier were as estimators and hyperparameters.


## 04_GridSearchCV
I have tried to find out the best combination of hyperparameters by grid search for RandomForestRegressor model.


## 05_SGDClassifier
The SGDClassifier is a linear classifier in scikit-learn that implements a stochastic gradient descent (SGD) algorithm for training.

I have tried to implement SGDClassifier on MNIST_784 dataset. Then applied PCA on the dataset with 95% variation ratio and implemented SGDClassifier on this dataset. It seems faster then the first model perspective of time.


## 06_DBSCAN
The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm is a density-based clustering algorithm used to group together data points that are close to each other in a given feature space.


## 07_TSNE
I have applied TSNE(T-Distributed Stochastic Neighbour Embedding) on MNIST 784 dataset then did plot. Then applied PCA, LDA(Linear Discriminant Analysis) and GaussianRandomProjection to plot.


## 08_Tensorflow
Implemented tensorlow.nn.convolution on an image and tried to find the shape of the output image.



## 09_Fully-Connected Neural Networks
An autoencoder is primarly used to be trained in an unsupervised fashion. This means the model can be trained just using inputs, and no labels. 

The autoencoder takes the image data, compress it to a small latent space and then reconstruct the image from the compressed latent space values. This is a great way to provide useful features using unsupervised data, for supervised models.



## 10_Convolution Neural Networks

Convolution neural networks (CNNs) have gained unparalleled popularity in computer vision tasks. A convolution neural network consists of several important components, such as,

* Convolution layers
* Pooling layers
* Dense layers (for final prediction)

We will use CNNs to learn the Cifar10 dataset. Like MNIST, Cifar10 is another popular dataset in computer vision. But it is a much more interesting dataset as you will see. Specifically, I will go through the following.

* How to do simple data exploration
* How to define a CNN model
* How to format data to suit model training
* How to train a model