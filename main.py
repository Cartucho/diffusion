import config as cfg
from unet_parts import *

import os
import glob
import math
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as trns
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np


class CatsDataset(Dataset):
    def __init__(self, dataset, im_format, transforms=None):
        self.transforms = transforms
        self.data = []
        for path_im in glob.glob(os.path.join(dataset, im_format)):
            self.data.append(path_im)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        im_path = self.data[idx]
        im = Image.open(im_path)
        if self.transforms:
            im = self.transforms(im)
        return im


class SinusoidalTimeEmbedding(nn.Module):
    """ ref: https://raw.githubusercontent.com/hojonathanho/diffusion/master/diffusion_tf/nn.py """
    def __init__(self, emb_dim, device):
        super().__init__()
        assert  emb_dim % 2 == 0
        half_dim = emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=cfg.DTYPE, device=device) * -emb)


    def forward(self, time_steps):
      emb = self.emb[:]
      emb = time_steps[:, None] * emb[None, :]
      emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=-1)
      return emb


class UNet(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        emb_dim = cfg.EMBEDDING_DIM
        self.time_emb = SinusoidalTimeEmbedding(emb_dim, device)

        self.input = (Conv(channels_in, 64, 3, 1))
        self.down1 = (Down(64, 128, emb_dim))
        self.down2 = (Down(128, 256, emb_dim))
        self.down3 = (Down(256, 512, emb_dim))
        self.down4 = (Down(512, 1024, emb_dim))
        self.up1 = (Up(1024, 512, emb_dim))
        self.up2 = (Up(512, 256, emb_dim))
        self.up3 = (Up(256, 128, emb_dim))
        self.up4 = (Up(128, 64, emb_dim))
        self.out = (Conv(64, channels_out, 1))

    def forward(self, x, t):
        t = t.flatten()
        t_emb = self.time_emb(t)
        x1 = self.input(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x = self.down4(x4, t_emb)
        x = self.up1(x, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        return self.out(x)


def get_index(vals, t):
    vals_t = vals.gather(-1, t)
    if len(t) == 1: # Single image
        return vals_t.reshape(1, 1, 1) # C,H,W
    return vals_t.reshape(len(t), 1, 1, 1) # B,C,H,W


def forward_diffusion(sqrt_alphas_bar, sqrt_one_minus_alphas_bar, x_0, t, device):
    noise = torch.randn_like(x_0).to(device)
    x_0 = x_0.to(device)
    sqrt_alphas_bar_t = get_index(sqrt_alphas_bar, t).to(device)
    sqrt_one_minus_alphas_bar_t = get_index(sqrt_one_minus_alphas_bar, t).to(device)
    x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise
    return x_t, noise


def loss_fn(sqrt_alphas_bar, sqrt_one_minus_alphas_bar, model, x_0, t, device):
    x_t, noise = forward_diffusion(sqrt_alphas_bar, sqrt_one_minus_alphas_bar, x_0, t, device)
    noise_pred = model(x_t, t)
    return nn.functional.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(betas, sqrt_one_minus_alphas_bar, sqrt_recip_alphas, posterior_variance, x, t):
    """
        Uses the trained model to predict the noise in the image and returns the denoised image. 
    """
    beta_t = get_index(betas, t)
    sqrt_one_minus_alphas_bar_t = get_index(sqrt_one_minus_alphas_bar, t)
    sqrt_recip_alphas_t = get_index(sqrt_recip_alphas, t)
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - beta_t * model(x, t) / sqrt_one_minus_alphas_bar_t
    )
    posterior_variance_t = get_index(posterior_variance, t)

    if t == 0:
        return model_mean # No added noise if we are in the last step
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad()
def sample_image(e, betas, sqrt_one_minus_alphas_bar, sqrt_recip_alphas, posterior_variance, device):
    """
        Sample / generate new image from noise
    """
    im_size = cfg.RESIZE[0]
    x = torch.randn((1, 3, im_size, im_size), device=device) # Image made of just random noise
    n_images = cfg.N_IM
    step_int = int(cfg.T / n_images)
    # Create pic, a stack of images from pure noise (right) to the generated image (left)
    width_total = im_size * n_images
    x_offset = width_total - im_size
    pic = Image.new('RGB', (width_total, im_size)) # The n_images are stacked together horizontally
    for i in range(0, cfg.T)[::-1]:
        t = torch.tensor([i], device=device, dtype=torch.int64)
        x = sample_timestep(betas, sqrt_one_minus_alphas_bar, sqrt_recip_alphas, posterior_variance, x, t)
        if i % step_int == 0:
            im = transforms_reversed(x[0]) # Here, we have a batch of 1, so we can use [0] to "de-batch"
            pic.paste(im, (x_offset, 0))
            x_offset -= im_size
    # Save the picture
    out_path = os.path.join(cfg.OUT_DIR, f"epoch_{e}.jpg")
    pic.save(out_path)


""" From PIL image to tensor """
transforms_custom = trns.Compose(
    [
        trns.RandomHorizontalFlip(),
        trns.Resize(size=cfg.RESIZE),
        trns.ToTensor(),  # Go from values [0, 255] into [0, 1]
        trns.Lambda(lambda x: (x * 2.) - 1) # Go from values [0, 1] into [-1, 1]
    ]
)
""" From tensor to PIL image """
transforms_reversed = trns.Compose(
    [
        trns.Lambda(lambda x: (x + 1.) / 2), # Go from values [-1, 1] into [0, 1]
        trns.Lambda(lambda x: x.permute(1, 2, 0)), # Re-order channels C,H,W -> H,W,C
        trns.Lambda(lambda x: x * 255.), # Go from values [0, 1] into [0, 255.]
        trns.Lambda(lambda x: x.cpu().numpy().astype(np.uint8)),
        trns.ToPILImage()
    ]
)


# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\tUsing {device} device")

if __name__ == "__main__":
    """ Load data """
    dataset = CatsDataset(dataset=cfg.DATA, im_format=cfg.DATA_FORMAT, transforms=transforms_custom)
    train_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    """ Forward diffusion """
    T = cfg.T
    betas = torch.linspace(cfg.BETA_START, cfg.BETA_END, T).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, axis=0)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
    """ Model """
    n_channels = dataset[0].shape[0]
    model = UNet(n_channels, n_channels).to(device)
    #print(model)
    """ Training"""
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    alphas_bar_prev = nn.functional.pad(alphas_bar[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)

    epoch_start = 0
    if not os.path.isdir(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    if not os.path.isdir(cfg.OUT_WEIGHTS):
        os.mkdir(cfg.OUT_WEIGHTS)
    else:
        if cfg.LOAD_WEIGHTS:
            path = os.path.join(cfg.OUT_WEIGHTS, cfg.WEIGHT_TO_LOAD)
            if os.path.isfile(path):
                print(f"Loading weights:{path}")
                model.load_state_dict(torch.load(path))
                epoch_start = int(path.split("_epoch_")[1].split(".pth")[0])
            else:
                print(f"Error: weights not found in {path}")
                exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
 
    assert cfg.SAVE_STEP < cfg.EPOCHS
    comment = f'batch={cfg.BATCH_SIZE}_lr={cfg.LR}_bstart={cfg.BETA_START}_bend={cfg.BETA_END}_T={cfg.T}'
    writer = SummaryWriter(comment=comment)
    loss_epoch = None
    with tqdm(range(epoch_start, cfg.EPOCHS), unit="epoch", leave=False, colour="GREEN") as tqdm_epoch:
        for epoch in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch [{epoch}/{cfg.EPOCHS}[ ")
            if loss_epoch:
                 tqdm_epoch.set_postfix(loss=loss_epoch) # report on tqdm the previous epoch loss
            loss_batches = []
            with tqdm(train_loader, unit="batch", leave=False, colour="CYAN") as tqdm_batches:
                for batch_i, batch in enumerate(tqdm_batches):
                    tqdm_batches.set_description(f"Batch [{batch_i+1}/{len(train_loader)}[ ")
                    optimizer.zero_grad()
                    t = torch.randint(0, cfg.T, (len(batch),), device=device, dtype=torch.int64) # sometimes len(batch) < cfg.BATCH_SIZE
                    loss = loss_fn(sqrt_alphas_bar, sqrt_one_minus_alphas_bar, model, batch, t, device)
                    loss.backward()
                    optimizer.step()
                    tqdm_batches.set_postfix(loss=loss.item()) # report on tqdm the batch loss
                    loss_batches.append(loss.item())
                loss_batches = np.array(loss_batches) # Convert to numpy array
                loss_epoch = np.average(loss_batches) # Measure epoch loss as the average of all batches
                writer.add_scalar("Loss/train", loss_epoch, epoch + 1) # report epoch loss on tensorboard
                if (epoch + 1) % cfg.SAVE_STEP == 0:
                    out_path = os.path.join(cfg.OUT_WEIGHTS, f"model_epoch_{epoch + 1}.pth")
                    torch.save(model.state_dict(), out_path)
                """ Sampling """
                sample_image(epoch + 1, betas, sqrt_one_minus_alphas_bar, sqrt_recip_alphas, posterior_variance, device)
    writer.flush()
    writer.close()
