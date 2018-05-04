from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys


def main():

    img_path = sys.argv[1]

    model = MobileNet()

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = decode_predictions(model.predict(x), top=3)[0]

    print(preds)


main()
