import os
import cv2
from torchvision import models, transforms
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import time


class FastSpriteCluster:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.vgg16_features = None

    def read_channels(self, image_type, sprite_file_name, orientation):
        sprite_front = cv2.imread(
            f"data/processed/{image_type}/{orientation}/{sprite_file_name}",
            cv2.IMREAD_UNCHANGED,
        )
        b, g, r, a = cv2.split(sprite_front)
        sprite_bgra = cv2.merge((b, g, r, a))
        sprite_bgra = cv2.cvtColor(sprite_bgra, cv2.COLOR_BGRA2RGBA)
        return b, g, r, a, sprite_bgra

    def vgg_features(self, image_type, orientation):
        if not self.vgg16_features:
            print("Loading VGG weights...")
            vgg16 = models.vgg16(pretrained=True)
            vgg16.eval()
            self.vgg16_features = vgg16.features
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize as per ImageNet
            ]
        )

        if not isinstance(image_type, list):
            image_type = [image_type]

        sprites_vgg_features = []
        sprites_list = []
        for type in image_type:
            print(f"Processing {type} images")
            sprites = os.listdir(f"data/processed/{type}/{orientation}/")
            for file in sprites:
                sprite = Image.open(
                    f"data/processed/{type}/{orientation}/" + file.split("/")[-1]
                ).convert("RGB")
                sprites_list.append((type, file.split("/")[-1], sprite))
                sprite = transform(sprite).unsqueeze(0)
                with torch.no_grad():
                    features = self.vgg16_features(sprite)
                sprites_vgg_features.append(features.flatten())
        return sprites_list, np.array(sprites_vgg_features)

    def flatten_sprites(self, image_type, orientation):
        sprites = os.listdir(f"data/processed/{image_type}/{orientation}/")
        sprite_images = []
        for file in sprites:
            _, _, _, _, sprite = self.read_channels(
                image_type, file.split("/")[-1], orientation
            )
            sprite_images.append(sprite.flatten())

        data = np.array(sprite_images)
        data = data / 255.0
        return sprite_images, data

    def find_k(self, flatten_array):
        num_clusters = 10
        inertia = []
        for k in range(1, num_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(flatten_array)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(12, 2))
        plt.plot(inertia)
        plt.show()

    def predict_clusters(self, flatten_array, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(flatten_array)
        print(Counter(kmeans.labels_))
        return kmeans.labels_

    def plot_clustered_sprites(
        self, cluster, labels=None, sprite_images_list=None, sprite_size=1.5
    ):
        if not isinstance(labels, np.ndarray):
            assert (
                len(self.labels) > 0
            ), "If no labels are passed as args, you should run the cluster method first"
            labels = self.labels
        if not isinstance(sprite_images_list, list):
            assert (
                len(self.sprite_images_list) > 0
            ), "If no sprite_images_list is passed as args, you should run the cluster method first"
            sprite_images_list = [x[2] for x in self.sprite_images_list]

        images_array = np.array(sprite_images_list)[labels == cluster]
        n = len(images_array)
        n_cols = int(np.ceil(n / 5)) if n > 5 else n
        n_rows = min(5, int(np.ceil(n / 5)) + 2)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(sprite_size * n_cols, sprite_size * n_rows)
        )
        row = 0
        col = 0
        for idx, s in enumerate(images_array):
            if (idx % n_cols == 0) & (idx > 0):
                row += 1
                col = 0
            axes[row, col].imshow(s.reshape(64, 64, -1))
            axes[row, col].axis("off")
            col += 1
        for i in range(idx + 1, n_rows * n_cols):
            if i % n_cols == 0:
                row += 1
                col = 0
            axes[row, col].axis("off")
            col += 1
        fig.suptitle(f"Cluster: {cluster}", fontsize=10)

    def cluster(self, image_type, orientation):
        self.sprite_images_list, vgg_a = self.vgg_features(image_type, orientation)
        self.find_k(vgg_a)
        k = int(input("Select the cluster k: "))
        print(f"Selected clusters: {str(k)}")
        self.labels = self.predict_clusters(vgg_a, k)
