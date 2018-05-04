import requests
import sys
import base64
import json


def encode_image(image_url):
    with open(image_url, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():

    if len(sys.argv) is not 2:
        print("Insufficient Arguments")
        return

    API_KEY = "AIzaSyAnLDwwZp1VuJ8QUrMP5i9lge3XNbkX4Qo"
    API_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

    IMAGE_URL = "https://lh3.googleusercontent.com/6x17rh1IYHNXl0fLWn4yVEj6dHugoUctuuMVIp018Qfo5iIRfoLzs2UsIhrlhVSbLC6Ywx3oPRATt_eZGqn1ZgPJXe-0-UBlkQgxH3Sge-EW46LJU7beyyn75kmA-86zkMtOFOXHJ7Tti7A0n1BEiSyiGTolOrcbI1LCEDq4-W7loSr9_GuCauYpiPktldhNsEDV104fMYRDVTKVuIBXbJvpUj0EbCX2RTf8fRLU288dXprjZb1SnmME6HyEfa8iwfnmuSIFvX_PJOq43ayVDqG0kZ0xRVBGQosk_W7Tw2RXV06bTxiJY3DMFel-hlLD_uiGttKcwtUJlGocMgDOqLBrJ_sqzbAJmQsIyI_V4E3Q46PMQRMzJMuMcS6huqAwQ7gD-_EBT0RNgdW4OfnE-irvagJgGNJV_92Ew1nQMT4eCzFibys7BDjpq80kT0qSZrETA9LqkEJ3ojDVG33c2W01a91GDUMjVmR8c43Hk55IoKHLrWWzazoqCwwOEQ_fzCgLgX7AH7U3DL6sLppRoBoUyQ55Ko2ZIVPDAqzr0Z1SZaXtlxXXAzBd-J9ob-sSAI5xWivssI1o7A5nadEfTlruIf9fqE6Bkfzr4mY=w1226-h1634-no"
    params = {"key": API_KEY}
    encoded_image = encode_image(sys.argv[1])

    data = {
        "requests": [
            {
                "image": {
                    "source": {
                        "imageUri": IMAGE_URL
                    }
                },
                "features": [
                    {
                        "type": "LABEL_DETECTION",
                        "maxResults": 10
                    }
                ]
            }
        ]
    }

    r = requests.post(url=API_ENDPOINT, params=params, json=data)

    result = json.loads(r.text)["responses"][0]["labelAnnotations"]

    for r in result:
        descr = r["description"]
        score = r["score"]
        print(f"{descr}: {score}")


main()
