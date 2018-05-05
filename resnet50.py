from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys


def main():

    img_path = sys.argv[1]

    model = ResNet50(weights="imagenet")

    test_image = image.load_img(img_path, target_size=(224, 224))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = preprocess_input(test_image)
    result = decode_predictions(model.predict(test_image), top=3)[0]


main()
