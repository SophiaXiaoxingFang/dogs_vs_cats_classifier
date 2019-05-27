# Define model
import sys
from keras import models
from keras import layers
from keras import optimizers

from matplotlib import pyplot as plt
import json
import numpy as np

LEARNING_RATE = 1e-4
EPOCH_NUM = 50
BATCH_SIZE = 32

def train_classifier(l_rate=LEARNING_RATE,epoch=EPOCH_NUM):

    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    validation_features = np.load('validation_features.npy')
    validation_labels = np.load('validation_labels.npy')

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(7, 7, 512)))
    model.add(layers.Dense(256, activation='relu', input_dim=(7 * 7 * 512)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    # Compile model
    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features, train_labels,
                        epochs=epoch,
                        batch_size=BATCH_SIZE,
                        validation_data=(validation_features, validation_labels))

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

    model_name = 'dogsvscats-translearning-{}-{}'.format(l_rate, epoch)
    model.save('{}.h5'.format(model_name))

    with open('{}-history.json'.format(model_name), 'w') as f:
        json.dump(history.history, f)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_classifier()
    elif len(sys.argv) == 3:
        train_classifier(float(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Invalid arguments input number of train_classifier!\n")