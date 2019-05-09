
# Importing the required libraries
import sys
import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm    # for interactive loading


# Constant definition: file path, image size
TRAIN_DIR = '.\\data\\train'
TEST_DIR = '.\\data\\test'
IMG_SIZE = 80


# Labelling the data-set
def label_img(img):
    word_label = img.split('.')[0]#[-3]
    # One hot encoder
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


# Processing the training data
def process_train_data(image_size=IMG_SIZE):
    # Creating an empty list where we should the store the training data
    training_data = []

   # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):
        # labeling the images
        label = label_img(img)

        path = os.path.join(TRAIN_DIR, img)

        # loading the image from the path and then converting them into gray-scale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (image_size, image_size))

        # final step-forming the training data list with numpy array of the images
        training_data.append([np.array(img), np.array(label)])

    # shuffling of the training data to preserve the random state of our data
    shuffle(training_data)

    # saving our trained data for further uses if required
    np.save('train_data.npy', training_data)
    return training_data


# Processing the given test data, similar with the prepossessing of training data expect for labeling
def process_test_data(image_size=IMG_SIZE):
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# Run the processing for obtaining the training and the testing data-set
if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_data = process_train_data()
        test_data = process_test_data()
    elif len(sys.argv) == 2:
        train_data = process_train_data(int(sys.argv[1]))
        test_data = process_test_data(int(sys.argv[1]))
    else:
        print("Invalid arguments input number of process_train_data and process_test_data!\n")



