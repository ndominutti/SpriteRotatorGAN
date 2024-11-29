import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
import socket
import psutil
import mlflow


class CustomImageDataset(Dataset):
    def __init__(self, input_files: list, target_file: list, transform):
        self.input_files = input_files
        self.target_file = target_file
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def read_img(self, img_path):
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        image = self.read_img(self.input_files[idx])
        label = self.read_img(self.target_file[idx])
        return image, label


def train_test_split(
    input_direction,
    target_direction,
    test_json_dir="../test_images.json",
    transform=None,
):
    """_summary_

    Args:
        test_json_dir (str): _description_
        directions (list): first direction is the X, second is the y

    Returns:
        _type_: _description_
    """
    with open(test_json_dir, "r") as f:
        test_images = json.load(f)

    X_train_paths = []
    y_train_paths = []
    X_test_paths = []
    y_test_paths = []
    for sprite_type in test_images.keys():
        X_path = f"../data/processed/{sprite_type}/{input_direction}"
        y_path = f"../data/processed/{sprite_type}/{target_direction}"

        sprites = os.listdir(X_path)
        train_idx = list(set(sprites).difference(set(test_images[sprite_type])))
        X_train_paths.extend([X_path + "/" + idx for idx in train_idx])
        y_train_paths.extend([y_path + "/" + idx for idx in train_idx])

        X_test_paths.extend([X_path + "/" + idx for idx in test_images[sprite_type]])
        y_test_paths.extend([y_path + "/" + idx for idx in test_images[sprite_type]])

    return CustomImageDataset(
        X_train_paths, y_train_paths, transform
    ), CustomImageDataset(X_test_paths, y_test_paths, transform=None)


def train_step(
    real_img,
    target_img,
    gen,
    disc,
    gen_opt,
    disc_opt,
    g_loss,
    d_loss,
    disc_mean_loss,
    gen_mean_loss,
    cur_step,
    writer,
):
    disc_opt.zero_grad()
    disc_loss = d_loss(real_img, gen, disc, target_img)
    disc_loss.backward(retain_graph=True)
    disc_opt.step()

    gen_opt.zero_grad()
    gen_loss, _ = g_loss(real_img, gen, disc, target_img)
    gen_loss.backward()
    gen_opt.step()

    disc_mean_loss += disc_loss.item()
    gen_mean_loss += gen_loss.item()

    writer.add_scalar("train_Gen_loss", gen_mean_loss, global_step=cur_step)
    writer.add_scalar("train_Disc_loss", disc_mean_loss, global_step=cur_step)

    return (
        gen,
        disc,
        gen_opt,
        disc_opt,
        gen_mean_loss / (cur_step + 1),
        disc_mean_loss / (cur_step + 1),
    )


def eval(
    dataloader: torch.utils.data.IterableDataset,
    gen,
    disc,
    g_loss,
    d_loss,
    device,
    writer,
    gstep,
    to_mlflow=False,
):
    gen_mean_loss = 0
    disc_mean_loss = 0
    cur_step = 0
    l1_mean_loss = 0
    fid = FrechetInceptionDistance(normalize=True)
    with torch.no_grad():
        for real_img, target_img in tqdm(dataloader):
            real_img = real_img.to(device)
            target_img = target_img.to(device)
            disc_loss = d_loss(real_img, gen, disc, target_img)
            gen_loss, recon_loss = g_loss(real_img, gen, disc, target_img)
            disc_mean_loss += disc_loss.item()
            gen_mean_loss += gen_loss.item()
            l1_mean_loss += recon_loss.item()

            fid.update(real_img[:, :3, :, :].to(torch.float64), real=True)
            fake_images = gen(real_img)
            norm_fake_images = (fake_images - fake_images.min()) / (
                fake_images.max() - fake_images.min()
            )
            fid.update(norm_fake_images[:, :3, :, :].to(torch.float64), real=False)
            cur_step += 1

    gen_mean_loss = gen_mean_loss / (cur_step + 1)
    disc_mean_loss = disc_mean_loss / (cur_step + 1)
    l1_mean_loss = l1_mean_loss / (cur_step + 1)
    fid_metric = fid.compute().item()

    if to_mlflow:
        metrics = {
            "eval_GenAdv_loss": gen_mean_loss,
            "eval_GenL1_loss": l1_mean_loss,
            "eval_Disc_loss": disc_mean_loss,
            "eval_GenFID_loss": fid_metric,
        }
        mlflow.log_metrics(metrics)
    else:
        writer.add_scalar("eval_GenAdv_loss", gen_mean_loss, global_step=gstep)
        writer.add_scalar("eval_GenL1_loss", l1_mean_loss, global_step=gstep)
        writer.add_scalar("eval_Disc_loss", disc_mean_loss, global_step=gstep)
        writer.add_scalar("eval_GenFID_loss", fid_metric, global_step=gstep)

    return (
        gen_mean_loss,
        disc_mean_loss,
        fid_metric,
        l1_mean_loss,
    )


def write_hardware_specs():
    machine_name = socket.gethostname()
    cpu_count = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024**3)
    memory_used = memory.used / (1024**3)
    memory_available = memory.available / (1024**3)
    mlflow.log_param("Machine Info", f"Machine Name: {machine_name}")
    mlflow.log_param("CPU Info", f"CPU Cores: {cpu_count}")
    mlflow.log_param(
        "Memory Info",
        f"Total Memory: {memory_total:.2f} GB, Used Memory: {memory_used:.2f} GB, Available Memory: {memory_available:.2f} GB",
    )
    gpu_info = "No GPU available"
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}, Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB\
            , Memory Cached: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB"
    mlflow.log_param("GPU Info", gpu_info)


def setup_mlflow(mlflow_experiment, tracking_uri):
    with open("../creds.json", "r") as f:
        creds = json.load(f)

    mlflow.set_tracking_uri(tracking_uri)
    os.environ["AWS_ACCESS_KEY_ID"] = creds["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["secret_access_key"]
    mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = creds["bucket_uri"]
    mlflow.set_experiment(mlflow_experiment)


def train(
    train_dataloader,
    test_dataloader,
    epochs,
    gen,
    disc,
    gen_opt,
    disc_opt,
    g_loss,
    d_loss,
    device,
    img_save_step: int,
    eval_step: int,
    writer_log_dir: SummaryWriter,
    mlflow_experiment: str,
    mlflow_tracking_uri="http://localhost:8080",
):
    initial_time = time()
    gen_mean_loss = 0
    disc_mean_loss = 0
    cur_step = 0
    saved_models = []
    writer = SummaryWriter(writer_log_dir)
    setup_mlflow(mlflow_experiment, mlflow_tracking_uri)
    with mlflow.start_run(run_name=writer_log_dir.split("/")[-1]):
        mlflow.log_param("Tensorboard_run", writer_log_dir)
        write_hardware_specs()
        for epoch in range(epochs):
            for real_img, target_img in tqdm(train_dataloader):
                real_img = real_img.to(device)
                target_img = target_img.to(device)

                gen, disc, gen_opt, disc_opt, gen_mean_loss, disc_mean_loss = (
                    train_step(
                        real_img,
                        target_img,
                        gen,
                        disc,
                        gen_opt,
                        disc_opt,
                        g_loss,
                        d_loss,
                        disc_mean_loss,
                        gen_mean_loss,
                        cur_step,
                        writer,
                    )
                )

                if cur_step % img_save_step == 0:
                    idx = np.random.choice(len(real_img))
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(real_img[idx].permute(1, 2, 0).detach().numpy())
                    axes[0].set_title("INPUT IMAGE")
                    axes[1].imshow(gen(real_img)[idx].permute(1, 2, 0).detach().numpy())
                    axes[1].set_title("GENERATED IMAGE")
                    axes[2].imshow(target_img[idx].permute(1, 2, 0).detach().numpy())
                    axes[2].set_title("TARGET IMAGE")

                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    img = Image.open(buf)
                    transform = transforms.ToTensor()
                    img_tensor = transform(img)
                    writer.add_image(
                        f"Generator output, step {cur_step}", img_tensor, 0
                    )
                    plt.close(fig)

                if cur_step % eval_step == 0:
                    print("> Evaluating...")
                    eval(
                        test_dataloader,
                        gen,
                        disc,
                        g_loss,
                        d_loss,
                        device,
                        writer,
                        cur_step,
                        to_mlflow=True,
                    )
                    print("> Resume training...")

                cur_step += 1

        print("> Final evaluation...")
        eval(
            test_dataloader,
            gen,
            disc,
            g_loss,
            d_loss,
            device,
            writer,
            cur_step,
        )

        mlflow.log_param(
            "Training info",
            f"Total training time: {str((time() - initial_time) / 60)} mins | Number of epochs: {epochs} | Total training steps: {cur_step}",
        )

        mlflow.pytorch.log_model(gen, "generator")
        mlflow.pytorch.log_model(disc, "discriminator")
        if not os.path.exists("mlflow_logs"):
            os.makedirs("mlflow_logs")
        torch.save(gen_opt.state_dict(), "mlflow_logs/gen_opt_weights.pth")
        mlflow.log_artifact("mlflow_logs/gen_opt_weights.pth")
        torch.save(disc_opt.state_dict(), "mlflow_logs/disc_opt_weights.pth")
        mlflow.log_artifact("mlflow_logs/disc_opt_weights.pth")


def load_model(model_path, gen, disc):
    checkpoint = torch.load(model_path)

    gen.load_state_dict(checkpoint["gen"])
    disc.load_state_dict(checkpoint["disc"])

    return gen, disc
