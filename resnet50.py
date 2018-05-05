from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys
import os


def predict(classifier, image_path):

    test_image = image.load_img(image_path, target_size=(224, 224))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = decode_predictions(classifier.predict(test_image), top=3)[0]

    return result


def main():

    img_path = sys.argv[1]

    model = ResNet50()

    correct = 0
    count = 0

    # for index, f in enumerate(os.listdir(img_path)):
    #     if index >= 10:
    #         break
    #     predictions = predict(model, os.path.join(img_path, f))
    #     count += 1
    #     for p in predictions:
    #         if "cat" in p[1]:
    #             correct += 1
    #             break

    predictions = predict(model, img_path)

    print(predictions)
    # print(correct / count)


main()
