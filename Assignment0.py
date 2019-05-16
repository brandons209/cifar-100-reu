from keras.layers.core import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


import argparse
from time import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-epochs", type=int, default=200, help="number of epochs to run for.")
parser.add_argument("-batch_size", type=int, default=128, help="batch size")
parser.add_argument("-model", type=str, default="inceptionv3", help="Model to use for training. Available: inceptionv3, InceptionResNetV2, ResNet50")
options = parser.parse_args()


batch_size = options.batch_size
num_classes = 100
epochs = options.epochs
model_name = options.model.lower()

if model_name == "inceptionv3":
    imported_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
    resize_flag = (75, 75)
elif model_name == "inceptionresnetv2":
    imported_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
    resize_flag = (75, 75)
elif model_name == "resnet50":
    imported_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    resize_flag = None

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if x_train.shape[1] == 1 else "s", *x_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if x_test.shape[1] == 1 else "s", *x_test.shape))

# input image dimensions
_, img_channels, img_rows, img_cols = x_train.shape

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#plt.imshow(np.reshape(x_train[0], (32,32,3)))
#plt.show(block=True)

def resize_data(dat, size):
    up_size = np.ndarray((dat.shape[0], size[0], size[1], 3), dtype=np.uint8)
    for image in range(dat.shape[0]):
        up_size[image] = cv2.resize(dat[image,:, :,:], size, interpolation=cv2.INTER_AREA)
    return up_size

if resize_flag:
    x_train = resize_data(x_train, resize_flag)
    x_test = resize_data(x_test, resize_flag)

print(x_train.shape[0])
print(x_test.shape[0])

## needs to incorpate size of image depending on model
def gen_subset(num_images, img_per_class, x_train, y_train):
    X_sub = np.ndarray((num_images, 75, 75, 3), dtype=np.float32)
    y_sub = []
    curr = 0

    for i in set(np.argmax(y_train, axis=1)):
        subi = x_train[np.argmax(y_train, axis=1) == i]
        rand_ind = np.random.choice(len(subi), img_per_class, replace=False)

        for ind in rand_ind:
            X_sub[curr, :] = subi[ind]
            y_sub.append(i)
            curr += 1
    y_sub = to_categorical(np.array(y_sub), 100)

    return X_sub, y_sub

print(x_train.shape)
print(x_test.shape)
"""
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
"""
# freeze
#for layer in inception_model.layers:
#    layer.trainable = False

x = imported_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=imported_model.input, outputs=x)

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
#aug = ImageDataGenerator(horizontal_flip=True)
filepath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
tensorboard = TensorBoard(log_dir="tensorboard_logs")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(patience=5, verbose=1)

#model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)

model.fit(x_train, y_train, batch_size=batch_size,
                    callbacks=[checkpoint, tensorboard, early],
                    epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test,  verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

"""
# a hackish transfer learning scenario - now use different labels
(z_train,), (z_test,) = cifar100.load_data(sources=['fine_labels'])
# convert class vectors to binary class matrices
Z_train = np_utils.to_categorical(z_train, num_classes)
Z_test = np_utils.to_categorical(z_test, num_classes)

model.fit(x_train, Z_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(x_test, Z_test))
score = model.evaluate(x_test, Z_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
"""
print(model.summary())
