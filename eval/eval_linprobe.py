# --------------------------------------------------------
# References:
# MAE: https://github.com/IcarusWizard/MAE
# --------------------------------------------------------
import sys
sys.path.append('../')

import glob
import math
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision
import utils
from icecream import ic
from model import MAE_ViT, ViT_Classifier
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np

# trade-off between speed and accuracy.
torch.set_float32_matmul_precision("medium")

def log_val_predictions(images, dataset_classes, labels, outputs, predicted, test_table, log_counter, num_images: int = 16):
  '''Log the predictions of the model on the validation dataset'''
  import wandb
  # obtain confidence scores for all classes
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()

  # convert log_labels to class names
  log_labels = np.array([dataset_classes[i] for i in log_labels])
  # convert log_preds to class names
  log_preds = np.array([dataset_classes[i] for i in log_preds])

  # adding ids based on the order of the images
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # add required info to data table:
    # id, image pixels, model's guess, true label, scores for all classes
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i.transpose(1, 2, 0)), p, l, *s)
    _id += 1
    if _id == num_images:
      break

def linprobe(cfg):
    # define the model, dataset and pca mode
    dataset_name = cfg["MAE"]["dataset"]
    pca_mode = cfg["MAE"]["pca_mode"]
    model_name = cfg["MAE"]["model_name"]
    run_name = '_' + dataset_name + '_' + pca_mode + '_' + time.strftime("%Y.%m.%d-%H.%M.00")
    folder_name = f"../model/{dataset_name}/{pca_mode}"

    # wandb logging
    wandb_log = cfg["logging"]["wandb_log"]
    wandb_project = cfg["logging"]["wandb_project"]
    wandb_run_name = cfg["logging"]["wandb_run_name"] + '_' + 'linprobe' + run_name

    batch_size = cfg["LINPROBE"]["batch_size"]
    load_batch_size = min(cfg["LINPROBE"]["max_device_batch_size"], batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # Load dataset
    dataset_name = cfg["MAE"]["dataset"]
    root_path = f"../data/{dataset_name}"
    
    # Transformation - These transformations are good for CIFAR-10, STL-10. 
    # For ImageNet, you may need to change the transformations. (You can use the commented transformations)
    transform_train = v2.Compose([
        v2.Resize((cfg["MAE"]["MODEL"]["image_size"], cfg["MAE"]["MODEL"]["image_size"])),
        # v2.RandomResizedCrop(cfg["MAE"]["MODEL"]["image_size"]),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # typically from ImageNet
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_val = v2.Compose([
        v2.Resize((cfg["MAE"]["MODEL"]["image_size"], cfg["MAE"]["MODEL"]["image_size"])),
        # v2.CenterCrop(cfg["MAE"]["MODEL"]["image_size"]),
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # typically from ImageNet
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset, val_dataset = utils.load_and_preprocess_images(root_path, dataset_name, transform_train, transform_val)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = utils.get_gpu()

    # Load model
    pretrain = True
    if os.path.exists(folder_name) and pretrain:
        # Search for .pt or .pth files in the folder
        model_files = glob.glob(os.path.join(folder_name, '*.pt')) + glob.glob(os.path.join(folder_name, '*.pth'))
        print(f"Model files found: {model_files}")
        if len(model_files) > 0:
            model_path = f"{folder_name}/{model_name}"
            # load the pre-trained model
            model = torch.load(model_path, map_location='cpu')
            print(f"Model loaded from {model_path}")
    else:
        # load the randomly initialized model
        model = MAE_ViT(
        image_size=cfg["MAE"]["MODEL"]["image_size"],
        patch_size=cfg["MAE"]["MODEL"]["patch_size"],
        emb_dim=cfg["MAE"]["MODEL"]["emb_dim"],
        encoder_layer=cfg["MAE"]["MODEL"]["encoder_layer"],
        encoder_head=cfg["MAE"]["MODEL"]["encoder_head"],
        decoder_layer=cfg["MAE"]["MODEL"]["decoder_layer"],
        decoder_head=cfg["MAE"]["MODEL"]["decoder_head"],
        mask_ratio=cfg["MAE"]["mask_ratio"],
        )
        print("Randomly initialized model loaded")
    
    model = ViT_Classifier(model.encoder, dropout_p = 0.0, num_classes=len(train_dataset.classes)).to(device)

    # freeze all but the head
    for name_, p in model.named_parameters():
        if "head" not in name_:
            p.requires_grad = False
    
    # model summary
    utils.summary(cfg, model, device, load_batch_size)

    compile_ = False
    if device == torch.device("cuda") and compile_:
        model = torch.compile(model) # * for faster training

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.SGD(model.head.parameters(), lr=cfg["LINPROBE"]["base_learning_rate"] * cfg["LINPROBE"]["batch_size"] / 256, momentum=0.9, weight_decay=cfg["LINPROBE"]["weight_decay"])
    lr_func = lambda epoch: min((epoch + 1) / (cfg["LINPROBE"]["warmup_epoch"] + 1e-8), 0.5 * (math.cos(epoch / cfg["LINPROBE"]["total_epoch"] * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=cfg)
        # wandb.watch(model, log="all", log_freq=25, log_graph=True)
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join('../logs', cfg["MAE"]["dataset"], 'lineprobe-cls'))

    best_val_acc = 0
    step_count = 0
    dataset_classes = val_dataset.classes
    optim.zero_grad()
    for e in range(cfg["LINPROBE"]["total_epoch"]):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        # W&B: Create a Table to store predictions for each test step
        if wandb_log:
            columns = ["id", "image", "predicted", "ground_truth"]
            for digit in range(len(train_dataset.classes)):
                columns.append(f"score_{dataset_classes[digit]}")
            test_table = wandb.Table(columns=columns)
        model.eval()
        log_counter = 0
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in iter(val_dataloader):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                _, predicted = torch.max(logits.data, 1)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
                if wandb_log and (log_counter < 4):
                    log_val_predictions(img, dataset_classes, label, logits, predicted, test_table, log_counter)
                    log_counter += 1
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            torch.save(model, f"{folder_name}/{cfg['LINPROBE']['output_model_path']}")

        if wandb_log:
            # Log the loss and learning rate
            wandb.log(
                {
                    # "epoch": e,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "train/acc": avg_train_acc,
                    "val/acc": avg_val_acc,
                    "lr": optim.param_groups[0]["lr"],
                },
                step=e,
            )
            # W&B: Log predictions table to wandb
            wandb.log({"val_predictions": test_table})
        else:
            writer.add_scalars('LINPROBE/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
            writer.add_scalars('LINPROBE/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)
            writer.add_scalar('lr', optim.param_groups[0]["lr"], global_step=e)
            # TODO: visualize the predicted images on val dataset for tensorboard
        
    # turn off the logging
    if wandb_log:
        # wandb.unwatch()
        wandb.finish()
    else:
        writer.close()



if __name__ == '__main__':
    # python eval_linprobe.py -c ../config/config_file.yaml
    parser = ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()

    print('Read Config File....')
    cfg = utils.load_yaml(args.config)
    ic(cfg)

    utils.setup_seed(cfg["seed"]) # set seed
    print("Start Linprobing....")
    linprobe(cfg)
    print("Linprobing Finished!")