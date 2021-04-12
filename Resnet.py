from keras.datasets import cifar10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.utils import  np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.input_layer import Input
from keras.initializers import glorot_uniform

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

NUM_CLASSES = 10

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

cols = 8
rows = 2
fig = plt.figure(figsize=(2*cols-1,2.5*rows-1))
for i in range(cols):
    for j in range(rows):
        rand_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i*rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[rand_index,:])
        ax.set_title(cifar10_classes[y_train[rand_index,0]])
plt.show()

X_train = x_train.astype('float32')
X_test = x_test.astype('float32')

X_train/=255
X_test/=255

Y_train = np_utils.to_categorical(y_train, len(cifar10_classes))
Y_test = np_utils.to_categorical(y_test, len(cifar10_classes))

x_val = X_train[:10000]
part_x_train = X_train[10000:]
y_val = Y_train[:10000]
part_y_train = Y_train[10000:]

print(x_val.shape)
print(part_x_train.shape)

gen = ImageDataGenerator(rotation_range=8, width_shift_range=.08, shear_range=.3, height_shift_range=.08, zoom_range=.08)
val_gen = ImageDataGenerator()
train_generator = gen.flow(part_x_train, part_y_train, batch_size=64)
val_generator = val_gen.flow(x_val, y_val, batch_size=64)

X_input = Input((32,32,3))
X = ZeroPadding2D(padding=(3,3), data_format=None)(X_input)
X = Conv2D(64,(7,7), strides=(2,2), name= 'conv1', kernel_initializer= glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
X = LeakyReLU(alpha=.01)(X)
X = MaxPooling2D((3,3), strides=(2,2)) (X)






X_shortcut = X

X = ZeroPadding2D(padding=(1,1), data_format=None)(X)
X = Conv2D(64, (3,3), strides = (1,1), name = 'conv2', kernel_initializer= glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3, name= 'bn_conv2')(X)
X = LeakyReLU(alpha=.01)(X)

X = ZeroPadding2D(padding=(1,1), data_format=None)(X)
X = Conv2D(64, (3,3), strides = (1,1), name='conv3', kernel_initializer= glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv3')(X)






X = Add()([X,X_shortcut])

X = LeakyReLU(alpha=.01)(X)

X = Flatten()(X)

X = Dense(NUM_CLASSES, activation='softmax', name='fc' + str(NUM_CLASSES), kernel_initializer= glorot_uniform(seed=0))(X)

model = Model(inputs = X_input, outputs = X, name='ResNet50')
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(), metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch= len(part_x_train)//64, epochs =10, validation_data=val_generator, validation_steps=len(x_val)//64)

score = model.evaluate(X_test, Y_test)
print()
print('Test Accuracy: ', score[1])