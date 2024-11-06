import cv2
import glob
import os
import ast
import argparse

orientation_coord = {
    "front": [3, 49, 54, 89],
    "left": [52, 97, 54, 89],
    "right": [100, 145, 54, 89],
    "back": [148, 193, 54, 89],
}


def saving_path_init(path):
    if not os.path.exists(path):
        print(f"Creating path {path}")
        os.makedirs(path)


def preprocess_pipeline(args):
    png_files = glob.glob(os.path.join(args.input_path, "*.png"))
    save_dir = (
        "/".join(png_files[0].replace("raw", "processed").split("/")[:-1])
        + "/"
        + args.orientation
    )
    saving_path_init(save_dir)
    for spath in png_files:
        name = spath.split("/")[-1]
        save_path = save_dir + "/" + name
        sprite = cv2.imread(spath, cv2.IMREAD_UNCHANGED)
        y_start, y_end, x_start, x_end = orientation_coord[args.orientation]
        sprite = sprite[y_start:y_end, x_start:x_end]
        sprite = cv2.resize(
            sprite, ast.literal_eval(args.output_shape), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(save_path, sprite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument(
        "--orientation", type=str, choices=["front", "left", "right", "back"]
    )
    parser.add_argument("--output-shape", type=str, default="(64, 64)")

    preprocess_pipeline(parser.parse_args())
