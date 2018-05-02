# Initialize Neural Network as Sequential
from keras.models import Sequential
# Perform Convolution Step on Training Images
# Use Max in Pooling Operation
# Flatten Arrays into a linear vector
# Dense is Fully Connected (Regular MLP)
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image Preprocessor to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image


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

    training_set.class_indices
    test_set = test_datagen.flow_from_directory('test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    classifier.fit_generator(training_set,
                             steps_per_epoch=4000,
                             epochs=5,
                             validation_data=test_set,
                             validation_steps=2000)

    return classifier


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


def main():
    classifier = create_model()
    classifier = train_classifier(classifier)

    classifier.save("model.h5")
    prediction = predict(classifier, "./Cat.jpg")

    print(prediction)
    f = open('prediction.txt', 'w')

    f.write(prediction)
    f.close()


main()
