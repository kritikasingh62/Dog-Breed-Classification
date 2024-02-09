<a name="br1"></a> 

**Indian Institute of Information Technology, Design and Manufacturing,**

**Jabalpur**

**CS314c Machine Learning Project (Literature Review)**

Dog breed classification

*Kritika Singh*

*2017132*

*Nikita Kushwaha Palak Mendiratta*

*2017162 2017175*

1\. **Dog Breed Identification (Research Paper)**

This project hopes to identify dog breeds from photographs. It is fine-grained

category trouble: First, we identify dog facial key points for every image using a

Convolution Neural Network. These keypoints are then used to extract capabilities

through SIFT descriptors and shade histograms. We then compare a variety of

different algorithms, which use these features to predict the breed of the canines

shown in the photo. The difficulties of identifying breeds because of variety are

compounded by the stylistic variations of images used inside the dataset, which

features dogs of the identical breed in numerous lightings and positions.[1]

● **Dog Breed Classification Using Part Localization**

This 2012 paper by Liu et. Al attempted dog breed identification using a comparable

approach. They first used an SVM regressor using grayscale SIFT descriptors as

features to isolate the face of the dog. Leverages a part localization algorithm, in

which a sliding window SVM detector using 1 SIFT greyscale descriptors is able to

classify their test dataset with an accuracy of about 90%. The classification set of

rules used focuses totally on the face of the dogs. This is partly because the face is

essentially a rigid object, simplifying the problem of comparing snapshots of various

dogs.[2]

**● Build Your First Computer Vision Project — Dog Breed Classification**

The Article demonstrates the step-by-step system on how computers interpret

photos and briefly goes over what a neural community, convolutional neural

network, and transfer learning are. Includes constructing and training of a CNN

architecture from scratch as well as observes transfer learning to vastly enhance

our model’s accuracy from 3.2% to 81%.

It's all related to the dog breed classification using CNN and Transfer learning.[3]

1



<a name="br2"></a> 

**● Using Convolutional Neural Networks to Classify Dog Breeds (Research paper**

**- II)**

In this paper by Hsu, David Stanford University, the task aims to apply a

convolutional neural network framework to teach and categorize dog breeds. It

approaches this first by using CNNs primarily based on LeNet and GoogLeNet

architectures.

The Stanford Dogs dataset is an open-access picture dataset of canine breeds.

There are a total of 120 breeds of dogs, with 20580 images in total, partitioned into

8580 test images, and 12000 training images. The image dataset comes with

annotations that mark out the bounding boxes which best includes the dog in the

images. Where the 3 breeds share almost all the same visible features however

belong to the best training. It is therefore thrilling to see how well CNNs can carry

out on handiest dog breeds, compared to labels from all classes of objects in the

everyday ImageNet.[4]

**● LeNet-5 Network Architectures**

This network is a convolutional neural network (CNN), these networks are built

upon 3 essential ideas: nearby receptive fields, shared weights, and spatial

subsampling. Local receptive fields with shared weights are the essence of the

convolutional layer LeNet made hand engineering capabilities redundant because

the network learns the high-quality internal representation from raw images

automatically. It simply has 7 layers, among which there are three convolutional

layers (C1, C3, and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 1 fully

connected layer (F6).[5]

**● LeNet CNN in Python on the MNIST digit recognition dataset.**

The purpose of this dataset is to categorize the handwritten digits 0-9. It’s also

pretty easy to get > 98% category accuracy in this dataset with minimal training

time, even on the CPU. LeNet is small and clean to understand — yet large enough

to offer interesting results. The article takes us through the basics of LeNet to

Model Building.[6]

**● Image data Preprocessing**

Help us understand the Keras API reference in the Preprocessing of Images and the

detailed explanation of each argument used to code. It is done before training the

model making it easier to extract key points for further analysis.[7]

2



<a name="br3"></a> 

**Improvement in our models**

**Importing Dataset**

We’ve used the Stanford Dogs Dataset with approximately 20,000 images of dogs of

120 breeds.

The data is imported from

<http://vision.stanford.edu/aditya86/ImageNetDogs/main.html>

After getting trained we will also be using validation images to fine-tune our

parameters, and testing the final model’s accuracy on test images.

**Creating a Le-Net CNN to Classify Dog Breeds**

We will be using the base Sequential model in designing our CNN networks. Due to

the small amount of data (only about 20K images i.e. only approximately 167 images

per breed), we did not have enough to train a deep neural network, thus we chose

to utilize a Convolutional Neural Network.

**LeNet- Network Architecture**

C1 layer - convolutional layer

S2 layer - pooling layer (downsampling layer)

C3 layer - convolutional layer

S4 layer - pooling layer (downsampling layer)

F6 layer - fully connected layer

**Training a CNN to Classify Dog Breeds**

We will start training the model we created (the no. of epochs is 300) and we see

that our validation loss constantly lowers, and our accuracy increases for each

epoch, signaling our model is learning.

**Model Evaluation**

The classification accuracy of each model is used to evaluate our models. The

models will predict the probability in each class of breed, then acquire the highest

probability class as the prediction of the class.

3



<a name="br4"></a> 

**References**

1\. Dog Breed Identification (Research Paper)

<https://web.stanford.edu/class/cs231a/prev_projects_2016/output%20(1).pdf>

2\. Dog Breed Classification Using Part Localization

<https://people.eecs.berkeley.edu/~kanazawa/papers/eccv2012_dog_final.pdf>

3\. Build Your First Computer Vision Project — Dog Breed Classification

[https://towardsdatascience.com/build-your-first-computer-vision-project-do](https://towardsdatascience.com/build-your-first-computer-vision-project-dog-breed-classification-a622d8fc691e)

[g-breed-classification-a622d8fc691e](https://towardsdatascience.com/build-your-first-computer-vision-project-dog-breed-classification-a622d8fc691e)

4\. Using Convolutional Neural Networks to Classify Dog Breeds Hsu, David

Stanford University

<http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf>

5\. LeNet-5 Architecture

[https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-](https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4)

[6fc3c59e6f4](https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4)

6\. LeNet - CNN in Python

[https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-ne](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)

[twork-in-python/](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)

7\. Keras Preprocessing Image <https://keras.io/api/preprocessing/image/>

4

