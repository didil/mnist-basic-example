import keras
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

test_labels

from keras import models
from keras import layers

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

simple_network = models.Sequential()
simple_network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
simple_network.add(layers.Dense(10, activation='softmax'))

simple_network.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

simple_network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = simple_network.evaluate(test_images, test_labels)

print('simple_network test_acc:', test_acc)

conv_network = models.Sequential()
conv_network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
conv_network.add(layers.MaxPooling2D((2, 2)))
conv_network.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_network.add(layers.MaxPooling2D((2, 2)))
conv_network.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_network.add(layers.Flatten())
conv_network.add(layers.Dense(64, activation='relu'))
conv_network.add(layers.Dense(10, activation='softmax'))

conv_network.compile(optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

conv_network.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = conv_network.evaluate(test_images, test_labels)

print('conv_network test_acc:', test_acc)


def predict_image(img_path, network, type):
    img = load_img(path=img_path, color_mode="grayscale", target_size=(28, 28, 1))

    image = img_to_array(img)
    if type == "simple":
        image = image.reshape(28 * 28)
    elif type == "conv":
        image = image.reshape((28, 28, 1))

    image = image.astype("float32") / 255.0
    images = np.expand_dims(image, axis=0)

    image_class = network.predict_classes(images)[0]

    return image_class, img, image


simple_results = []
conv_results = []
for img_path in ["./6.png", "./2.png", "./3.png"]:
    image_class, img, image = predict_image(img_path, simple_network, "simple")
    simple_results.append(image_class)

    image_class, img, image = predict_image(img_path, conv_network, "conv")
    conv_results.append(image_class)

print("simple_network:")
print("real: 6 , 2 , 3")
print("predicted:", simple_results)

print("conv_network:")
print("real: 6 , 2 , 3")
print("predicted:", conv_results)
