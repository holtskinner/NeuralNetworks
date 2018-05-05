from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys
import os


def predict(classifier, image_path):

    test_image = image.load_img(image_path, target_size=(224, 224))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = preprocess_input(test_image)
    result = decode_predictions(classifier.predict(test_image), top=1)[0]

    return result[0]


def main():

    img_path = sys.argv[1]

    model = MobileNet()

    correct = 0
    count = 0

    for index, f in enumerate(os.listdir(img_path)):
        if index >= 100:
            break

        p = predict(model, os.path.join(img_path, f))

        count += 1

        if "cat" in p[1] or p[1] == "tabby" or p[1] == "lynx":
            correct += 1

    print(correct / count)


main()
