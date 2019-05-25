import sys
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from matplotlib import pyplot as plt

from training import IMG_SIZE,CHANNELS,LEARNING_RATE,EPOCH_NUM

TEST_DIR = '../data/test/'

def prep_data(images,image_size):
    count = len(images)
    X = np.ndarray((count, image_size, image_size, CHANNELS), dtype=np.float32)

    for i, image_file in enumerate(images):
        img = image.load_img(image_file, target_size=(image_size, image_size))
        X[i] = image.img_to_array(img)
        X[i] /= 255.
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))
    return X

def test_classifier(image_size=IMG_SIZE,l_rate=LEARNING_RATE,epoch=EPOCH_NUM):
    # load the model
    model_name = 'dogsvscats-keras-da-{}-{}-{}'.format(image_size, l_rate, epoch)
    model = load_model('{}.h5'.format(model_name))
    model.compile(loss='binary_crossentropy',
                 optimizer=optimizers.RMSprop(lr=l_rate),
                 metrics=['accuracy'])

    test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

    X_test = prep_data(test_images,image_size)
    predictions = model.predict(X_test)

    # Select 20 figures to show the result
    fig = plt.figure()
    for i in range(0,20):
        y = fig.add_subplot(4, 5, i + 1)
        if predictions[i, 0] >= 0.5:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        y.imshow(image.array_to_img(X_test[i]))
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

    # output the csv file for all test data_set
    with open('submission_file.csv', 'w') as f:
        f.write('id,label\n')

    with open('submission_file.csv', 'a') as f:
        for i, path in enumerate(test_images):
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            if predictions[i, 0] >= 0.5:
                f.write('{},{}\n'.format(name, 1))
            else:
                f.write('{},{}\n'.format(name, 0))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_classifier()
    elif len(sys.argv) == 4:
        test_classifier(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
    else:
        print("Invalid arguments input number of test_classifier!\n")