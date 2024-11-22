import os
import json
import cv2
import numpy as np

with open("../test_images.json", "r") as f:
    test_images = json.load(f)


def train_test_generation(directions: list):
    """_summary_

    Args:
        test_images (dict): _description_
        directions (list): first direction is the X, second is the y

    Returns:
        _type_: _description_
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for d in directions:
        for sprite_type, v in test_images.items():
            sprites = os.listdir(f"../data/processed/{sprite_type}/{d}/")
            for s in sprites:
                sprite_file_name = s.split("/")[-1]
                sprite_front = cv2.imread(
                    f"../data/processed/{sprite_type}/{d}/{sprite_file_name}",
                    cv2.IMREAD_UNCHANGED,
                )
                b, g, r, a = cv2.split(sprite_front)
                sprite_bgra = cv2.merge((b, g, r, a))
                sprite_bgra = cv2.cvtColor(sprite_bgra, cv2.COLOR_BGRA2RGBA)
                if d == directions[0]:
                    if s in v:
                        X_test.append(sprite_bgra)
                    else:
                        X_train.append(sprite_bgra)
                else:
                    if s in v:
                        y_test.append(sprite_bgra)
                    else:
                        y_train.append(sprite_bgra)
    return X_train, X_test, y_train, y_test
