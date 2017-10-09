import tensorflow

import matplotlib
matplotlib.use("Agg")

import math
import numpy as np
from os import listdir, read
from os.path import isfile, join
import pandas
import os
from scipy import ndimage

labels_file = 'data/captcha_labels.csv'

# Number of  characters of output
n_output = 2
# Number of character classes
n_classes = 57

width = 60
height = 60
batch_size=256
epochs=400


csv_labels=pandas.read_csv(labels_file, na_filter=False)

# load all files into memory
X_train = []
y_train = []

X_test = []
y_test = []

train_to_test = 0.1

for index, row in csv_labels.iterrows():
    filename=row['path']
    if isfile(filename):
        image_data = ndimage.imread(filename)
        captcha_text = row['inputted_captcha']
        if captcha_text != captcha_text:
            print('Error loading {}'.format(captcha_text))
            raise Exception()
        if index > train_to_test * len(csv_labels):
            X_train.append(image_data)
            y_train.append(captcha_text)
        else:
            X_test.append(image_data)
            y_test.append(captcha_text)

train_samples=len(X_train)
test_samples=len(X_test)
print('Train samples: {}'.format(len(X_train)))
print('Test samples:  {}'.format(len(X_test)))

# input = 60x160x3  (RGB image)
# output = 5x57     (5 chars one-hot encoded)

# Normalize the data features to the variable X_normalized
print("Input shape: {}".format(np.shape(X_train)))

def normalize(x):
    a = -0.5
    b = 0.5
    return a + ( (x * (b - a) )/ 255 )

X_normalized = np.array([normalize(x) for x in X_train])
assert math.isclose(np.min(X_normalized), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_normalized), 0.5, abs_tol=1e-5), 'The range of the training data is: {} to {}.  It must be -0.5 to 0.5'.format(np.min(X_normalized), np.max(X_normalized))
print('Normalization OK 👍.')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

y_temp = np.array([[i for i in x] for x in y_train])
# transform to integer
lbl_enc=LabelEncoder()
one_hot_enc=OneHotEncoder()
y_int = lbl_enc.fit_transform(y_temp.ravel()).reshape(*y_temp.shape)
# transform to binary
y_one_hot = one_hot_enc.fit_transform(y_int).toarray()

assert np.shape(y_one_hot) == (train_samples, n_output * n_classes)
print('One-hot encoding OK 👍')

print(lbl_enc.classes_)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="valid", input_shape=(height, width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding="valid"))
model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))

model.add(Conv2D(64, (3, 3), padding="valid"))
model.add(Dropout(.5))
model.add(Activation('relu')) 

model.add(Convolution2D(256, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))
model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(n_output*n_classes*2))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(n_output * n_classes))
model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

assert model.layers[0].input_shape == (None, height, width, 3), 'First layer input shape is wrong, it should be (60,160,3)'
print(np.shape(X_normalized))
print(np.shape(y_one_hot))
history = model.fit(X_normalized, y_one_hot, batch_size=batch_size, epochs=epochs, validation_split=0.2)

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

X_test = np.array([normalize(x) for x in X_test])
y_temp = np.array([[i for i in x] for x in y_test])
# transform to integer
y_int = lbl_enc.transform(y_temp.ravel()).reshape(*y_temp.shape)
# transform to binary
y_one_hot = one_hot_enc.transform(y_int).toarray()

ev = model.evaluate(X_test, y_one_hot, batch_size=32, verbose=1, sample_weight=None)
print()
print(model.metrics_names)
print(ev)

for i in range(10):
    print("Label:",y_test[i])
    pred = model.predict(X_test[i].reshape((1,height, width,3)))
    reshaped = pred.reshape((n_output, -1))
    print([lbl_enc.classes_[np.argmax(x)] for x in reshaped])
