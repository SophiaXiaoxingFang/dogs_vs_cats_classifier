import sys
import os
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from keras.applications import VGG16
from matplotlib import pyplot as plt

from training import LEARNING_RATE,EPOCH_NUM

IMG_SIZE = 224 # Default input size for VGG16
CHANNELS = 3

TEST_DIR = 'small_data_set/test/'

def test_classifier(l_rate=LEARNING_RATE,epoch=EPOCH_NUM):
    # load the model
    model_name = 'dogsvscats-translearning-{}-{}'.format(l_rate, epoch)
    model = load_model('{}.h5'.format(model_name))
    model.compile(loss='binary_crossentropy',
                 optimizer=optimizers.RMSprop(lr=l_rate),
                 metrics=['accuracy'])

    test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

    fig = plt.figure()
    for i, image_file in enumerate(test_images):
        img = image.load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
        features = conv_base.predict(img_tensor.reshape(1, IMG_SIZE, IMG_SIZE, CHANNELS))
        prediction = model.predict(features)

        y = fig.add_subplot(4, 5, i + 1)
        if prediction >= 0.5:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        y.imshow(img_tensor)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_classifier()
    elif len(sys.argv) == 3:
        test_classifier(float(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Invalid arguments input number of test_classifier!\n")