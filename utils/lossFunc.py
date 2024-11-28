import torch
import torch.nn as nn


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

        # Reconstruction loss: L1(y, G(x))
        recon_loss = self.l1_loss(G_out, target_image)

        # Combined loss
        combined_loss = adv_loss + self.lambda_reg * recon_loss
        return combined_loss, recon_loss


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
