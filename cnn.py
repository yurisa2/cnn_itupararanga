#  Description: This program classifies images

# Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Print the data type of x_train
print(type(x_train))
# Print the data type of y_train
print(type(y_train))
# Print the data type of x_test
print(type(x_test))
# Print the data type of y_test
print(type(y_test))

# Get the shape of x_train
print('x_train shape:', x_train.shape) # 4D array 50,000 rows 32x32 pixel image with depth = 3 visible wave lenghts (RGB)
# Get the shape of y_train
print('y_train shape:', y_train.shape) # 2D array 50,000 rows and 1 column
# Get the shape of x_train
print('x_test shape:', x_test.shape) # 4D array 10,000 rows 32x32 pixel image with depth = 3 visible wave lenghts (RGB)
# Get the shape of y_train
print('y_test shape:', y_test.shape) # 2D array 10,000 rows and 1 column

# Take a look at the first image (at index=0) in the training data set as a numpy array
# This shows the image as a series of pixel values
x_train[0][0][0]

# Show the image as an image instead of a series of pixel values using matplotlib
import matplotlib.pyplot as plt
img = plt.imshow(x_train[0])

# Print the label of the image, NOTE: the number 6 = frog
# 0 = airplane
# 1 = automobile
# 2 = bird
# 3 = cat
# 4 = deer
# 5 = dog
# 6 = frog
# 7 = horse
# 8 = ship
# 9 = truck
print('The label is:', y_train[0])

# One-Hot Encoding
# Convert the labels into a set of 10 numbers to input into the neural network
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print all of the new labels in the training data set
print(y_train_one_hot)

# Print an example of the new labels, NOTE: The label 6 = [0,0,0,0,0,0,1,0,0,0]
print('The one hot label is:', y_train_one_hot[0])

# normalize the pixels in the images to be a value between 0 and 1 , they are normally values between 0 and 255
# doing this will help the neural network.
x_train = x_train / 255
x_test = x_test / 255

#  Build The CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential() # Create the architecture

# Convolution layer to extract features from the input image, and create 32 ReLu
# 5x5 convolved features/layers aka feature map.
# Note:You must input the input shape only in this first layer.
#  number of output channels or convolution filters = 32
#  number of rows in the convolution kernel
#  number of cols in the convolution kernel
#  input shape 32x32 RGB image, so spacially it's 3-Dimensional
#  activation function Rectifier Linear Unit aka (ReLu)
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))


# Pooling layer with a 2x2 filter to get the max element from the convolved features ,
# this reduces the dimensionality by half e.g. 16x16, aka sub sampling
# Note: the default for stride is the pool_size
model.add(MaxPooling2D(pool_size=(2, 2)))



# 2nd Convolution layer with 64 channels
model.add(Conv2D(64, (5, 5), activation='relu'))

# Adding second Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening, Flattens the input. Does not affect the batch size.
# (Flattening occurs when you reduce all layers to one background layer),
# this makes the image a linear array or 1D Array or 1D Vector to
# feed into or connect with the neural network
model.add(Flatten())
model.add(Dense(1000, activation='relu')) #  a layer with 1000 neurons and activation function ReLu
model.add(Dense(10, activation='softmax')) # a layer with 10 output neurons for each label using softmax activation function

model.compile(loss='categorical_crossentropy', #  loss function used for classes that are greater than 2)
              optimizer='adam',
              metrics=['accuracy'])

# Batch: Total number of training examples present in a single batch
# Epoch:The number of iterations when an ENTIRE dataset is passed forward and
#       backward through the neural network only ONCE.
# Fit: Another word for train

# NOTE: We don't need to use validation_data, so we didn't have to split the data
# into a validation sets. We just put in 0.2 and this splits the data 20% for us.
hist = model.fit(x_train, y_train_one_hot,
           batch_size=256, epochs=10, validation_split=0.3 )

# Get the models accuracy
model.evaluate(x_test, y_test_one_hot)[1]
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# Visualize the models accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Load the data
# from google.colab import files #  Use to load data on Google Colab
# uploaded = files.upload() #  Use to load data on Google Colab

import os
os.chdir("C:/Users/Administrator/Documents/rPDI/artigo/sat/python")

my_image = plt.imread("cat.4014.jpg") # Read in the image (3, 14, 20)
# Show the uploaded image
img = plt.imshow(my_image)


# Resize & Show the image
import skimage
from skimage.transform import resize
my_image_resized = resize(my_image, (32,32,3)) # resize the image to 32x32 pixel with depth = 3
img = plt.imshow(my_image_resized) # show new image

# Get the probabilities for each class
# model.predict function is expecting an array, so we will use np.array to make this transformation on the image
import numpy as np
probabilities = model.predict(np.array( [my_image_resized,] ))

# Show the probability for each class
probabilities

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])

# To save this model
model.save('my_model.h5')

# To load this model
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
