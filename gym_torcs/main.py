import keras as kr
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split

from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from PIL import Image


np.random.seed(0)
INPUT_SHAPE = (64, 64, 3)
PATH_TO_CSV = "."

def load_image(image_path):
    return mpimg.imread(image_path)

def load_data(config):
    data_df = pd.read_csv(os.path.join(config.data_dir, 'db.csv'))  # TODO: do rename this to 'driving log' or something else more informative than 'db'
    X = data_df['image'].values
    y = data_df['ctrl'].values

    # TODO: what is this 'random state' business?
    X_train, X_validate, y_train, y_validate = train_test_split(X,y, test_size=config.test_size, random_state =0, shuffle=True)

    print('-' * 30)
    print("Len Train:", len(X_train))
    print("Len Val:", len(X_validate))
    print('-' * 30)
    return X_train, X_validate, y_train, y_validate

def random_flip(image, ctrl):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        ctrl = -ctrl
    return image, ctrl

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augment(image, ctrl):
    image, ctrl = random_flip(image, ctrl)
    image = random_brightness(image)
    return image, ctrl

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    return image
    # image = rgb2yuv(image)
    # return image

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def batch_generator(image_paths, control_data, batch_size, is_training):
    images = np.empty([batch_size, *INPUT_SHAPE])
    ctrls = np.empty(batch_size)
    while True:
        i=0
        for index in np.random.permutation(image_paths.shape[0]):
            # get the images and controls
            # # TODO: set the augment rate as a config parameter.
            # if is_training and np.random.rand() < 0.6:
            #     image, ste
            # else:
            image = load_image(image_paths[index])
            ctrl = control_data[index]

            # TODO: make augment rate a parameter in config
            if is_training:
                image, ctrl = augment(image, ctrl)
            images[i] = preprocess(image)
            ctrls[i] = ctrl

            i+=1
            if i == batch_size:
                break

        yield images, ctrls


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2), input_shape=INPUT_SHAPE))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Flatten())

    model.add(Dense(1000, activation='elu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Dropout(args.keep_prob))
    model.add(Dense(100, activation='elu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Dropout(args.keep_prob))
    model.add(Dense(50, activation='elu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Dropout(args.keep_prob))
    model.add(Dense(10, activation='elu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Dense(1))
    model.summary()

    if args.weights:
        print("Loading model weights from %s" % args.weights)
        model.load_weights(args.weights)

    return model


def train_model(model, config, X_train, X_validate, y_train, y_validate):
    """
    Trains model, use validation set to display performance while training.
    """

    checkpoint = kr.callbacks.ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=config.save_best_only,
                                 mode = 'auto')

    model.compile(loss='mean_squared_error', optimizer=kr.optimizers.Adam(lr=config.learning_rate))

    tensorboard = TensorBoard(log_dir ="./logs/{}".format(time.time()))

    try:
        model.fit_generator(batch_generator(X_train, y_train, config.batch_size, True),
                            config.samples_per_epoch,
                            config.nb_epoch,
                            max_q_size=1,
                            validation_data = batch_generator(X_validate, y_validate, config.batch_size, False),
                            nb_val_samples = 500,
                            callbacks =[checkpoint, tensorboard],
                            verbose = 1)
    except KeyboardInterrupt:
        pass
    finally:
        model_json = model.to_json()
        with open("model_def.json", 'w') as f:
            f.write(model_json)
        model.save_weights("weights.h5")



def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():
    parser = argparse.ArgumentParser(description = "SDCC Behavioral Cloning on TORCS")
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.05)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=100)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=500)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=64)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    parser.add_argument('-w', help='saved weights to load',               dest='weights',            type=str,   default='')
    parser.add_argument('-v', help='visualize training samples', dest='visualize', action='store_true')
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():  #  TODO: Vars returns __dict__ attribute.
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)

    if args.visualize:
        batch = next(batch_generator(data[0], data[2], args.batch_size, True))
        for i in range(10):
            img = Image.fromarray(np.uint8(batch[0][i]))
            img.show()
    else:
        model = build_model(args)
        train_model(model, args, *data)

if __name__ == "__main__":
    main()
