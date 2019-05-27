from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16 # convolutional base
import numpy as np

TRAIN_NUM = 2000
VALIDATION_NUM = 400
TEST_NUM = 12500
BATCH_SIZE = 32
IMG_SIZE = 224 # Default input size for VGG16
CHANNELS = 3

train_data_dir = 'small_data_set/train'
validation_data_dir = 'small_data_set/validation'

# Extract features
def extract_bottleneck_features(directory, sample_count):

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    datagen = ImageDataGenerator(rescale=1. / 255)

    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    return features, labels


if __name__ == '__main__':
    train_features, train_labels = extract_bottleneck_features(train_data_dir, TRAIN_NUM)  # Agree with our small dataset size
    validation_features, validation_labels = extract_bottleneck_features(validation_data_dir, VALIDATION_NUM)

    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    np.save('validation_features.npy', validation_features)
    np.save('validation_labels.npy', validation_labels)

