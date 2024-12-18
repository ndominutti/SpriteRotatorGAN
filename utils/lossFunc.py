import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import gc


class GenLoss(nn.Module):
    def __init__(self, lambda_reg):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_reg = lambda_reg

    def forward(self, input, gen, disc, target_image):
        G_out = gen(input)
        D_out = disc(G_out, input)
        # Adversarial loss: -E[log(D(G(x)))]
        adv_loss = self.bce_loss(D_out, torch.ones_like(D_out))
        self.adv_loss = adv_loss
        # Reconstruction loss: L1(y, G(x))
        recon_loss = self.l1_loss(G_out, target_image)

        # Combined loss
        combined_loss = adv_loss + self.lambda_reg * recon_loss
        return combined_loss, adv_loss, recon_loss


class GenLossWithPerceptual(GenLoss):
    def __init__(self, lambda_reg, lambda_perc, device, features_layer="Conv2_2"):
        super().__init__(lambda_reg)
        layers_dict = {
            "Conv1_2": 4,
            "Conv2_2": 9,
            "Conv3_3": 15,
            "Conv4_3": 23,
            "Conv5_3": 30,
        }
        self.lambda_perc = lambda_perc
        vgg16 = models.vgg16(pretrained=True)
        self.perceptual_model = nn.Sequential(
            *list(vgg16.features)[0 : layers_dict[features_layer]]
        )
        for param in self.perceptual_model.parameters():
            param.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, input, gen, disc, target_image):
        combined_loss, recon_loss = super().forward(input, gen, disc, target_image)
        G_out = gen(input)
        G_out = F.interpolate(
            G_out, size=(224, 224), mode="bilinear", align_corners=False
        )[:, :3, :, :]
        G_out = (G_out - self.mean) / self.std
        target_image_vgg = F.interpolate(
            target_image, size=(224, 224), mode="bilinear", align_corners=False
        )[:, :3, :, :]
        target_image_vgg = (target_image_vgg - self.mean) / self.std
        with torch.no_grad():
            target_features = self.perceptual_model(target_image_vgg)
            generated_features = self.perceptual_model(G_out)
        perceptual_loss = self.l1_loss(generated_features, target_features)
        combined_loss += self.lambda_perc * perceptual_loss
        return combined_loss, super().adv_loss, recon_loss, perceptual_loss


class GenLossWithSober(nn.Module):
    def __init__(self, lambda_reg):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_reg = lambda_reg

    def forward(self, input, gen, disc, target_image):
        G_out = gen(input)
        D_out = disc(G_out, input)
        # Adversarial loss: -E[log(D(G(x)))]
        adv_loss = self.bce_loss(D_out, torch.ones_like(D_out))
        self.adv_loss = adv_loss
        # Reconstruction loss: L1(y, G(x))
        recon_loss = self.l1_loss(G_out, target_image)

        # Combined loss
        combined_loss = adv_loss + self.lambda_reg * recon_loss
        return combined_loss, adv_loss, recon_loss


class DiscLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, gen, disc, target_imag):
        G_out = gen(input)
        D_fake = disc(G_out.detach(), input)
        D_real = disc(target_imag, input)
        fake_loss = self.bce_loss(D_fake, torch.zeros_like(D_fake))
        real_loss = self.bce_loss(D_real, torch.ones_like(D_real))
        return (fake_loss + real_loss) / 2
