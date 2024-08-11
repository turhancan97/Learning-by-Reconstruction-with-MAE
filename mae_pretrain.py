import math
import os
import time
from argparse import ArgumentParser

import torch
import torchvision
from einops import rearrange
from icecream import ic
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import utils
from model import *

# trade-off between speed and accuracy.
torch.set_float32_matmul_precision("medium")

def train(cfg):
    # wandb logging
    wandb_log = cfg["logging"]["wandb_log"]
    wandb_project = cfg["logging"]["wandb_project"]
    wandb_run_name = cfg["logging"]["wandb_run_name"] + time.strftime("%Y.%m.%d-%H.%M.%S")

    batch_size = cfg["MAE"]["batch_size"]
    load_batch_size = min(cfg["MAE"]["max_device_batch_size"], batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size
    
    # Load dataset
    dataset_name = cfg["MAE"]["dataset"]
    root_path = f"data/{dataset_name}"
    
    # Common transformation
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    
    train_dataset, val_dataset = utils.load_and_preprocess_images(root_path, dataset_name, transform)

    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
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

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=cfg)
        wandb.watch(model, log="all", log_freq=100, log_graph=True)
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join('logs', cfg["MAE"]["dataset"], 'mae-pretrain'))

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

        if wandb_log:
            # Log the loss and learning rate
            wandb.log(
                {
                    # "epoch": e,
                    "train/loss": avg_loss,
                    # "val/loss": losses['val'],
                    "lr": optim.param_groups[0]["lr"],
                },
                step=e,
            )
            
            # Add image to wandb to visualize
            add_image = wandb.Image((img + 1) / 2, caption=f"Epoch:{e}")
            wandb.log({"mae_image": add_image,},step=e)
        else:
            writer.add_scalar('train/loss', avg_loss, global_step=e)
            writer.add_scalar('lr', optim.param_groups[0]["lr"], global_step=e)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
    
    # turn off the logging
    if wandb_log:
        wandb.unwatch()
        wandb.finish()
    else:
        writer.close()
    
    dataset_name = cfg["MAE"]["dataset"]
    pca_mode = cfg["MAE"]["pca_mode"]
    model_name = cfg["MAE"]["model_name"]
    folder_name = f"model/{dataset_name}/{pca_mode}"

    # create a folder to save the model if it does not exist
    os.makedirs(folder_name, exist_ok=True)
    # save model
    torch.save(model, f"{folder_name}/{model_name}")

if __name__ == '__main__':
    # python mae_pretrain.py -c config/config_file.yaml
    parser = ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()

    print('Read Config File....')
    cfg = utils.load_yaml(args.config)
    ic(cfg)

    utils.setup_seed(cfg["seed"]) # set seed
    print("Start Training....")
    train(cfg)
    print("Training Finished....")