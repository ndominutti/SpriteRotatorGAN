import json
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import argparse
import os
from tqdm import tqdm

np.random.seed(0)


def apply_gs_transformer(Ximage, yimage):
    if np.random.rand() > 0.5:
        transformer = T.Grayscale(1)
        r, g, b, Xa = Ximage.split()
        rgb_Ximage = Image.merge("RGB", (r, g, b))
        r, g, b, Ya = yimage.split()
        rgb_yimage = Image.merge("RGB", (r, g, b))
        Xgs = transformer(rgb_Ximage)
        Xgs = Image.merge("RGBA", (Xgs, Xgs, Xgs, Xa))
        ygs = transformer(rgb_yimage)
        ygs = Image.merge("RGBA", (ygs, ygs, ygs, Ya))
        return Xgs, ygs
    else:
        return Ximage, yimage


def apply_gb_transformer(Ximage, yimage):
    if np.random.rand() > 0.5:
        transformer = T.GaussianBlur(kernel_size=(1, 5), sigma=(3, 4))
        Xgs = transformer(Ximage)
        ygs = transformer(yimage)
        return Xgs, ygs
    else:
        return Ximage, yimage


def apply_cj_transformer(Ximage, yimage, transformer):
    if np.random.rand() > 0.5:
        r, g, b, Xa = Ximage.split()
        rgb_Ximage = Image.merge("RGB", (r, g, b))
        r, g, b, Ya = yimage.split()
        rgb_yimage = Image.merge("RGB", (r, g, b))
        _, brightness, contrast, saturation, hue = transformer.get_params(
            transformer.brightness,
            transformer.contrast,
            transformer.saturation,
            transformer.hue,
        )

        Xgs = T.functional.adjust_brightness(rgb_Ximage, brightness)
        Xgs = T.functional.adjust_contrast(Xgs, contrast)
        Xgs = T.functional.adjust_saturation(Xgs, saturation)
        Xgs = T.functional.adjust_hue(Xgs, hue)

        ygs = T.functional.adjust_brightness(rgb_yimage, brightness)
        ygs = T.functional.adjust_contrast(ygs, contrast)
        ygs = T.functional.adjust_saturation(ygs, saturation)
        ygs = T.functional.adjust_hue(ygs, hue)

        r, g, b = Xgs.split()
        Xgs = Image.merge("RGBA", (r, g, b, Xa))
        r, g, b = ygs.split()
        ygs = Image.merge("RGBA", (r, g, b, Ya))
        return Xgs, ygs
    else:
        return Ximage, yimage


def apply_pad_transformer(Ximage, yimage):
    if np.random.rand() > 0.5:
        padding = int(np.random.uniform(5, 30))
        transformer = T.Pad(padding=padding, fill=0, padding_mode="constant")
        Xgs = transformer(Ximage).resize((64, 64))
        ygs = transformer(yimage).resize((64, 64))
        return Xgs, ygs
    else:
        return Ximage, yimage


def augment_data(args):
    """Augment training images (avoid augmenting test ones)

    Args:
        test_json_dir (str): _description_

    Returns:
        _type_: _description_
    """
    with open(args.test_json_dir, "r") as f:
        test_images = json.load(f)

    counter = 0
    for sprite_type in tqdm(test_images.keys()):
        X_path = f"../data/processed/{sprite_type}/{args.input_orient}"
        y_path = f"../data/processed/{sprite_type}/{args.target_orient}"
        sprites = os.listdir(X_path)
        train_idx = list(set(sprites).difference(set(test_images[sprite_type])))
        cj_transformer = T.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(3),
            saturation=(0.3, 1.5),
            hue=(-0.1, 0.1),
        )

        for idx in train_idx:
            Xpath = X_path + "/" + idx
            ypaht = y_path + "/" + idx
            Ximage = Image.open(Xpath).convert("RGBA")
            yimage = Image.open(ypaht).convert("RGBA")
            for i in range(args.n_iter):
                X, y = apply_gs_transformer(Ximage, yimage)
                X, y = apply_gb_transformer(X, y)
                X, y = apply_pad_transformer(X, y)
                X, y = apply_cj_transformer(X, y, cj_transformer)
                if X != Ximage:
                    counter += 1
                    x_aug_path = (
                        X_path + "/" + idx.replace(".png", f"_augmented_{i}.png")
                    )
                    y_aug_path = (
                        y_path + "/" + idx.replace(".png", f"_augmented_{i}.png")
                    )
                    X.save(x_aug_path)
                    y.save(y_aug_path)

    print(f"A total of {counter} images were augmented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-orient", type=str, choices=["front", "left", "right", "back"]
    )
    parser.add_argument(
        "--target-orient", type=str, choices=["front", "left", "right", "back"]
    )
    parser.add_argument("--n-iter", type=int)
    parser.add_argument("--test-json-dir", type=str, default="../test_images.json")
    augment_data(parser.parse_args())
