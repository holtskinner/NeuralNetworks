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

    # IMAGE_URL = "https://lh3.googleusercontent.com/O8AdmYP5bZ0PpgDVgOwsk6qkIZ1_NCjLHm2EtDhLoXZ9ZN8R53-Jc5dsJ39xlqVSoTU5Ee0a7aaeDQgxYWA2c31uWsva_uRBKdZDmDmqem85wSfMnpThrzFSZlsqj7SYPPHG_elJOC0JIfLLvK_2cSf_tLKuTUzpS8-wEsZL3HklaCVZ4V3LgIKGfUXGhf21mr9RxUewq8J-hwDhKyC4CUqSvqy52NksC-AGJCy6rWdvEzwlC_gEXcZDPAxBue1_oGVSoougJ-ZzxyMgITJrJK248ScOZU-UhuocpS-7AT1OcQD_w9S2JnNFQTUfn7UrrGB8z-0w_WPy0vmNvrw9QLt93N0Bj08jtVE6kAZsR43tV2EhBOEPw75hh0mNAXFuPUJnPpmpXcNqJgK1xEblGXhuNCBfAOTCELps5KjPLbbm26xF_pu9IkZaOtAhBnWBK5bbwCr-yDtvKitmHB-GxRNW5HD5Zzf5axmuIIRhgVmhtCms607sGFkV943SJ3JrUmYl7jCMUfH097_9B3I6eFeOQDkhn3gS7L5ZMAJprXFrHTaFGUU63tSoXqoyJHe79zrwkbeiXxPM8zICcQPhAojgfq4tlmLXRQCaJmKy=w1530-h2040-no"
    params = {"key": API_KEY}

    data = {
        "requests": []
    }

    img_dir = sys.argv[1]
    for i, f in enumerate(os.listdir(img_dir)):

        if i == 100:
            break
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
