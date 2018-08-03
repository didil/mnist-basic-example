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

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)


print('test_acc:', test_acc)

from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import matplotlib.pyplot as plt

def predict_image(img_path):
    img = load_img(path=img_path,color_mode = "grayscale",target_size=(28,28,1))

    image = img_to_array(img)
    image = image.reshape(28*28)
    image = image.astype("float32") / 255.0

    images = np.expand_dims(image, axis=0)

    image_class = network.predict_classes(images)[0]

    return image_class,img,image

results = []
for img_path in ["./6.png", "./2.png", "./3.png"]:
	image_class,img,image = predict_image(img_path)
	results.append(image_class)

#plt.imshow(img)

print("real: 6 , 2 , 3")
print("predicted:", results)

