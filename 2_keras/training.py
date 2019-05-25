import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers

from matplotlib import pyplot as plt
import json


IMG_SIZE = 150
LEARNING_RATE = 1e-4
EPOCH_NUM = 50

TRAIN_NUM = 2000
VALIDATION_NUM = 400
BATCH_SIZE = 32
CHANNELS = 3

train_data_dir = 'small_data_set/train'
validation_data_dir = 'small_data_set/validation'

def train_classifier(image_size=IMG_SIZE,l_rate=LEARNING_RATE,epoch=EPOCH_NUM):
    if K.image_data_format() == 'channels_first':
        input_shape = (CHANNELS, image_size, image_size)
    else:
        input_shape = (image_size, image_size, CHANNELS)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizers.RMSprop(lr=l_rate),
                 metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=TRAIN_NUM//BATCH_SIZE,
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=VALIDATION_NUM//BATCH_SIZE,
        verbose=2)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    model_name = 'dogsvscats-keras-da-{}-{}-{}'.format(image_size, l_rate, epoch)
    model.save('{}.h5'.format(model_name))

    with open('{}-history.json'.format(model_name), 'w') as f:
        json.dump(history.history, f)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_classifier()
    elif len(sys.argv) == 4:
        train_classifier(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
    else:
        print("Invalid arguments input number of train_classifier!\n")