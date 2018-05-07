from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Sequential, load_model, Model
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import sys
from keras.preprocessing.image import ImageDataGenerator


def train_classifier(classifier):

    # Creates extra training data by flipping, rotating images
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Pull all images from training_set directory
    training_set = train_datagen.flow_from_directory('training_set',
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('test_set',
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='binary')

    # Actual Training step
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch=8000,
                                       epochs=25,
                                       validation_data=test_set,
                                       validation_steps=2000)

    return classifier, history


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])


def main():

    img_path = sys.argv[1]

    base_model = ResNet50(include_top=False, weights="imagenet")
    model = add_new_last_layer(base_model, 2)
    setup_to_transfer_learn(model, base_model)

    model, history = train_classifier(model)

    model.save("resnet.h5")

    test_image = image.load_img(img_path, target_size=(224, 224))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = preprocess_input(test_image)
    result = decode_predictions(model.predict(test_image), top=3)[0]


main()
