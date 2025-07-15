import random
import time
from typing import Iterable

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils import register_plugin


def set_deterministic(seed=42):
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@register_plugin("optim", "adamw")
def build_adamw(model, optim_cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["reg_w"]
    )


@register_plugin("engine", "train_engine")
def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
):
    step = 0
    epoch_loss = 0

    model.train()
    for batch_data in data_loader:
        optimizer.zero_grad()
        batch_data = batch_data.cuda()
        batch_loss, pred, mask = model(
            batch_data, mask_ratio=cfg["MODEL"]["mask_ratio"]
        )
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
        # log
        if step % 100 == 0:
            print(f"Step [{step}], Loss: {batch_loss.item():.4f}")
        step += 1
    return epoch_loss / step


@register_plugin("engine", "my_train_engine")
def my_train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    epoch: int,
    logger: SummaryWriter,
):
    step = 0
    epoch_loss = 0

    model.train()
    for i, batch_data in enumerate(data_loader):
        optimizer.zero_grad()
        batch_data = batch_data.cuda()
        batch_loss, pred, mask = model(
            batch_data, mask_ratio=cfg["MODEL"]["mask_ratio"]
        )
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
        # log
        if step % 100 == 0:
            print(f"Step [{step}], Loss: {batch_loss.item():.4f}")
            epoch_1000x = int((i / len(data_loader) + epoch) * 1000)
            # log images
            n = min(8, batch_data.shape[0])
            original_imgs = batch_data[:n]
            patched_imgs = model.patchify(original_imgs)
            masked_imgs = patched_imgs * (1 - mask[:n, :, None])
            mixed_imgs = masked_imgs + pred[:n] * mask[:n, :, None]
            masked_imgs = model.unpatchify(masked_imgs)
            mixed_imgs = model.unpatchify(mixed_imgs)
            recon_imgs = model.unpatchify(pred[:n])
            grid = torchvision.utils.make_grid(
                torch.cat([original_imgs, masked_imgs, mixed_imgs, recon_imgs], dim=0),
                nrow=n,
                padding=0,
                normalize=True,
                scale_each=True,
            )
            logger.add_image(
                "train_images",
                grid[0],  # grid is stacked RGB, extract greyscale again
                epoch_1000x,
                dataformats="HW",
            )
        step += 1
    return epoch_loss / step
