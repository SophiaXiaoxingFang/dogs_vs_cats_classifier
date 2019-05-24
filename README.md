# Dogs_vs_Cats

## Introduction
The project is to implement a binary image classifier to tell a picture is Dog or Cat. The origin of this project is from a [Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats) .
In this repository, different approaches are tried to solve the problem:

  * [TFlearn, basic CNN](./tflearn/README.md)
  * [Keras, data augmentation](./keras_data_augmentation/README.md) 


## Prepare the data-set

The data-set can been obtained from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). 

The training data contains 25 000 labeled images of dogs and cats, each category has 12 500 images accordingly. The test data contains 12 500 unlabeled images.

Download the train.zip and test1.zip files, unzip them, and put the training data in (**./data/train/**) folder, and the testing data in the (**./data/test/**) folder.


## Reference
https://www.geeksforgeeks.org/image-classifier-using-cnn/
