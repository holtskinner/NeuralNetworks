# Initialize Neural Network as Sequential
from keras.models import Sequential, load_model
# Perform Convolution Step on Training Images
# Use Max in Pooling Operation
# Flatten Arrays into a linear vector
# Dense is Fully Connected (Regular MLP)
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image Preprocessor to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image
import sys
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def create_model():

    # Instansiate Network
    classifier = Sequential()

    # Convolution Layer 32 Filters 3x3 each, each image 64*64 RGB Relu Function
    classifier.add(
        Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Pooling Layer to reduce size of image
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    # Hidden Fully Connected Layer
    classifier.add(Dense(units=128, activation='relu'))

    # Output Layer Gives Binary Output (Cat or Dog)
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Optimizer - Use Stochastic Gradient Decent, Use Cross Entropy Loss Function, Measure Accuracy
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


def train_classifier(classifier):

    # Creates extra training data by flipping, rotating images
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    history = classifier.fit_generator(training_set,
                                       steps_per_epoch=8000,
                                       epochs=25,
                                       validation_data=test_set,
                                       validation_steps=2000)

    return classifier, history


def predict(classifier, image_path):

    test_image = image.load_img(image_path, target_size=(64, 64))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return prediction


def graph_model():
    lines = tuple(open("nohup.out", "r"))

    loss = np.zeros(len(lines))
    acc = np.zeros(len(lines))

    for i, c in enumerate(lines):
        digs = np.array(re.findall("\d+\.\d+", c))
        if len(digs) >= 2:
            loss[i] = digs[0]
            acc[i] = digs[1]

    acc = acc[5:20005]
    loss = loss[5:20005]

    for i in range(len(acc)):
        if acc[i] == 0:
            acc[i] = acc[i - 1]
        if loss[i] == 0 or loss[i] >= .9:
            loss[i] = loss[i - 1]

    plt.plot(acc, "b--", loss, "g--")
    plt.grid(True)
    plt.savefig("clusters.jpg")

    acc = np.reshape(acc, (5, 4000))
    loss = np.reshape(loss, (5, 4000))

    print(acc)
    print(loss)


def main():

    # img_path = sys.argv[1]
    # classifier = load_model("model.h5")

    classifier = create_model()
    classifier, history_callback = train_classifier(classifier)

    classifier.save("model.h5")

    loss_history = history_callback.history["loss"]
    np_loss_history = np.array(loss_history)
    np.savetxt("loss_history.txt", loss_history, delimiter=",")

    acc_history = history_callback.history["acc"]
    np_acc_history = np.array(acc_history)
    np.savetxt("acc_history.txt", acc_history, delimiter=",")

    # correct = 0
    # count = 0
    # for f in os.listdir(img_path):
    #     prediction = predict(classifier, os.path.join(img_path, f))
    #     if prediction == "cat":
    #         correct += 1
    #     count += 1

    # print(correct / count)
    # f = open('prediction.txt', 'w')

    # f.write(prediction)
    # f.close()


main()
# graph_model()
