import math
import os
from argparse import ArgumentParser

import torch
import torchvision
from einops import rearrange
from icecream import ic
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import utils
from model import *

# trade-off between speed and accuracy.
torch.set_float32_matmul_precision("medium")

if __name__ == '__main__':
    # python mae_pretrain.py -c config/config_file.yaml
    parser = ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()

    print('Read Config File....')
    cfg = utils.load_yaml(args.config)
    ic(cfg)

    utils.setup_seed(cfg["seed"]) # set seed

    batch_size = cfg["MAE"]["batch_size"]
    load_batch_size = min(cfg["MAE"]["max_device_batch_size"], batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = utils.get_gpu()

    model = MAE_ViT(
        image_size=cfg["MAE"]["MODEL"]["image_size"],
        patch_size=cfg["MAE"]["MODEL"]["patch_size"],
        emb_dim=cfg["MAE"]["MODEL"]["emb_dim"],
        encoder_layer=cfg["MAE"]["MODEL"]["encoder_layer"],
        encoder_head=cfg["MAE"]["MODEL"]["encoder_head"],
        decoder_layer=cfg["MAE"]["MODEL"]["decoder_layer"],
        decoder_head=cfg["MAE"]["MODEL"]["decoder_head"],
        mask_ratio=cfg["MAE"]["mask_ratio"],
    ).to(device)

    if device == torch.device("cuda"):
        model = torch.compile(model) # * for faster training
    
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["MAE"]["base_learning_rate"] * cfg["MAE"]["batch_size"] / 256, betas=(0.9, 0.95), weight_decay=cfg["MAE"]["weight_decay"])
    lr_func = lambda epoch: min((epoch + 1) / (cfg["MAE"]["warmup_epoch"] + 1e-8), 0.5 * (math.cos(epoch / cfg["MAE"]["total_epoch"] * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(cfg["MAE"]["total_epoch"]):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / cfg["MAE"]["mask_ratio"]
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model, cfg["MAE"]["model_path"])