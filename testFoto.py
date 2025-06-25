# testFoto.py

import os
import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


def check_image(image):
    height, width, channel = image.shape
    if abs(width / height - 4 / 3) > 0.1:
        return False
    return True


def test_image(image_path, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    frame = cv2.imread(image_path)
    if not check_image(frame):
        print("Invalid image aspect ratio")
        return None

    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    label = np.argmax(prediction)
    value = prediction[0][label] / 2

    if label == 1:
        if value > 0.996:
            result_text = "RealFace Score: {:.2f}".format(value)
        else:
            result_text = "SuspectFace Score: {:.2f}".format(value)
    else:
        result_text = "FakeFace Score: {:.2f}".format(value)

    print(result_text)
    print("Prediction cost {:.2f} s".format(test_speed))

    return label, value


if __name__ == "__main__":
    import argparse
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="model_lib used to test")
    parser.add_argument("--image_path", type=str, default="", help="image path to test")
    args = parser.parse_args()
    test_image(args.image_path, args.model_dir, args.device_id)
