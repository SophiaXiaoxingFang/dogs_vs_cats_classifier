# Image Classifier using CNN

# Importing the required libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for interactive loading
from cnn_model import build_cnn_model

IMG_SIZE = 80

# Building up CNN
def train_test_classifier(image_size=IMG_SIZE,lr=1e-3,epoch=5):
    model = build_cnn_model(image_size, lr)
    MODEL_NAME = 'dogsvscats-6conv-basic-{}-{}-{}.model'.format(image_size, lr, epoch)

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    else:
        # Loading the training data-set after pre-processing
        # and splitting the testing data and validation data
        train_data = np.load('train_data.npy')
        train = train_data[:-500]
        validation = train_data[-500:]

        # Setting up the features and lables
        # x-Features & y-Labels
        train_x = np.array([i[0] for i in train]).reshape(-1, image_size, image_size, 1)
        train_y = [i[1] for i in train]
        validation_x = np.array([i[0] for i in validation]).reshape(-1, image_size, image_size, 1)
        validation_y = [i[1] for i in validation]

        # Training the CNN model with the data-sets
        model.fit({'input': train_x}, {'targets': train_y}, n_epoch=epoch,
                            validation_set=({'input': validation_x}, {'targets': validation_y}),
                            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
        model.save(MODEL_NAME)

    # Testing the data
    # Loading the testing data-set after pre-processing
    test_data = np.load('test_data.npy')

    # Select 20 figures to show the result
    fig = plt.figure()
    for num, data in enumerate(test_data[:20]):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(4, 5, num + 1)
        orig = img_data
        data = img_data.reshape(image_size, image_size, 1)

        model_out = model.predict([data])[0]  # cat: [1, 0], dog: [0, 1]

        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

    # output the csv file for all test data_set
    with open('submission_file.csv', 'w') as f:
        f.write('id,label\n')

    with open('submission_file.csv', 'a') as f:
        for data in tqdm(test_data):
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(image_size, image_size, 1)
            model_out = model.predict([data])[0]
            # f.write('{},{}\n'.format(img_num, model_out[1]))
            if np.argmax(model_out) == 1:
                label = 1  # 'Dog'
            else:
                label = 0  # 'Cat'
            f.write('{},{}\n'.format(img_num, label))


# Run the two functions for training and testing the classifier
if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_test_classifier()
    elif len(sys.argv) == 4:
        train_test_classifier(int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]))
    else:
        print("Invalid arguments input number of train_test_classifier!\n")


