# Image Classifier using CNN

# Importing the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm    # for interactive loading
from cnn_model import build_cnn_model
from cnn_model import MODEL_NAME
from preprocessing import IMG_SIZE

# Building up CNN
model = build_cnn_model()

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
    train_x = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    train_y = [i[1] for i in train]
    validation_x = np.array([i[0] for i in validation]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    validation_y = [i[1] for i in validation]

    # Training the CNN model with the data-sets
    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=5,
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
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]# cat: [1, 0], dog: [0, 1]

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
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        # f.write('{},{}\n'.format(img_num, model_out[1]))
        if np.argmax(model_out) == 1:
            label = 1 #'Dog'
        else:
            label = 0 #'Cat'
        f.write('{},{}\n'.format(img_num, label))
