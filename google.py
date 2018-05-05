import requests
import sys
import base64
import json
import os


def encode_image(image_url):
    with open(image_url, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():

    if len(sys.argv) is not 2:
        print("Insufficient Arguments")
        return

    API_KEY = "AIzaSyAnLDwwZp1VuJ8QUrMP5i9lge3XNbkX4Qo"
    API_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

    params = {"key": API_KEY}

    data = {
        "requests": []
    }

    img_dir = sys.argv[1]
    for i, f in enumerate(os.listdir(img_dir)):

        encoded_image = encode_image(os.path.join(img_dir, f))
        data["requests"].append({
            "image": {
                "content": encoded_image
            },
            "features": [
                {
                    "type": "LABEL_DETECTION",
                    "maxResults": 1
                }
            ]
        })

        if i % 16 == 0:

            r = requests.post(url=API_ENDPOINT, params=params, json=data)

            result = json.loads(r.text)["responses"]

            for i in range(len(result)):
                r = result[i]["labelAnnotations"][0]
                descr = r["description"]
                score = r["score"]
                print(f"{descr}: {score}")

            data["requests"] = []


# main()
