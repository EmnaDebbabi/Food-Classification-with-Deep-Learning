# Food-Classification-with-Deep-Learning
# Food Classification with Deep Learning
## Introduction
### CNN in Computer Vision
###### What is Computer Vision?
Computer Vision is an interdisciplinary field of science that aims to make computers process, analyze images and videos and extract details in the same way a human mind does. Earlier Computer Vision was meant only to mimic human visual systems until we realized how AI can augment its applications and vice versa. We may also not realize this every day but we are being assisted by the applications of Computer Vision in automotive, retail, banking and financial services, healthcare, etc.

###### How CNNs Work?
Most of the Computer Vision tasks are surrounded around CNN architectures, as the basis of most of the problems is to classify an image into known labels. Algorithms for object detection like SSD(single shot multi-box detection) and YOLO(You Only Look Once) are also built around CNN.
</br> <img src="images/1.png" width="" height="">

## Project Description
We present a CNN-based dish detection and score estimation system, running on iOS or android. The proposed app can estimate the score after detecting dishes from the video stream captured from the built-in camera of a Phone. For the deep learning framework, we use Keras (backend TensorFlow) to train and convert the model for CoreML to use it.

###### What is Tensorflow?
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

###### what is Keras?
Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks.

</br> <img src="images/2.jpg" width="" height="">
## Dataset
We used Food-11 image dataset found in Kaggle <a href="https://www.kaggle.com/vermaavi/food11"> Kaggle Dataset</a></br>

* There are 3 splits in this dataset:
  evaluation </br>
  training </br>
  validation </br>
 * Each split contains 11 categories of food: </br>
  Bread </br>
  Dairy product </br>
  Dessert </br>
  Egg </br>
  Fried food </br>
  Meat </br>
  Noodles-Pasta </br>
  Rice </br>
  Seafood </br>
  Soup </br>
  Vegetable-Fruit </br>
## Approach
We built a system recently for the purpose of experimenting with Deep Learning. The key components are a single NVIDIA Tesla K80 GPU w/12 GB of memory that can be used up to 12 hours continuously offered by Google Colab,25 gb of system RAM.It is running 64-bit Ubuntu 19.04 and using the jupyter notebook Python 3.6 distribution.

## Results
After fine-tuning a pre-trained efficientnet model, I was able to achieve about 86.99% Top-1 Accuracy on the test set using a single crop per item.But when it came to prediction there is a noticed loss.
</br> * We have been able to achieve more accurate results with: </br>
  InceptionV3 </br>
  ResNet200 </br>
  WISeR </br>
</br> <img src="images/3.png" width="" height="">

## Thoughts and future improvement
We have spent about 2 months on and off building this project, trying to train dozens of models and exploring various areas such as multiprocessing for faster image augmentation. This is a cleaned up version of the notebook that contains my best performing model as of Sep 14, 2020.</br>
If you have more ressources (in term of ram and GPU) you can replace the dataset with another one which is richer(like FOOD-101 for example) and other different models to get better results. </br>
In the future, We would like to learn how to: </br>

* How to handle larger than RAM datasets in a performant way?
* How to avoid Loading a large amount of data into memory?
* Saving the data into h5py file for out of band processing?
* Improving multiprocessing image augmentation? </br>
Please get in touch if you have any ideas! </br
## Ressources:
I was inspired by this Keras blog post: <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html" >Building powerful image classification models using very little data </a></br>
and a related script I found on github that belongs to Keras too: <a href="https://keras.io/guides/transfer_learning/">keras-finetuning</a></br>
