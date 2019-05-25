# Dogs_vs_Cats

## Introduction
The project is to implement a binary image classifier to tell a picture is Dog or Cat. The origin of this project is from a [Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats) .
In this repository, different approaches are tried to solve the problem:

  * [TFlearn, basic CNN](./1_tflearn/README.md), a basic approach using `TFlearn` and full data-set. 
  * [Keras, data augmentation](./2_keras/README.md) , concerning small scale data-set (e.g., for each category, 1000 images for training, 200 images for validation), using data augmentation in `keras.preprocessing.image.ImageDataGenerator` class.

Please check out the code and readme file in the corresponding folders.

The project should be run using Python interpreter. It has been tested in PyCharm IDE as well as Command Prompt in Windows 10. 
For using the code, clone the files in this repository and unzip them in one folder.  

## Prepare the data-set

The data-set can been obtained from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). 

The training data contains 25 000 labeled images of dogs and cats, each category has 12 500 images accordingly. The test data contains 12 500 unlabeled images.

Download the train.zip and test1.zip files, unzip them, and put the training data in (**./data/train/**) folder, and the testing data in the (**./data/test/**) folder.


## Reference
https://www.geeksforgeeks.org/image-classifier-using-cnn/

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
