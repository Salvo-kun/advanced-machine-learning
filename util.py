
import torch
import shutil
import logging
from typing import Type, List
from argparse import Namespace

from cosface_loss import MarginCosineProduct
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.grl_dataset import GrlDataset
from model import network
import multiprocessing

def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(state: dict, is_best: bool, output_folder: str,
                    ckpt_filename: str = "last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state["model_state_dict"], f"{output_folder}/best_model.pth")


def resume_train(args: Namespace, output_folder: str, model: torch.nn.Module,
                 model_optimizer: Type[torch.optim.Optimizer], classifiers: List[MarginCosineProduct],
                 classifiers_optimizers: List[Type[torch.optim.Optimizer]]):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {args.resume_train}")
    checkpoint = torch.load(args.resume_train)
    start_epoch_num = checkpoint["epoch_num"]

    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    model = model.to(args.device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert args.groups_num == len(classifiers) == len(classifiers_optimizers) == \
        len(checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        (f"{args.groups_num}, {len(classifiers)}, {len(classifiers_optimizers)}, "
         f"{len(checkpoint['classifiers_state_dict'])}, {len(checkpoint['optimizers_state_dict'])}")

    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(args.device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        # Move classifiers back to CPU to save some GPU memory
        c = c.cpu()

    best_val_recall1 = checkpoint["best_val_recall1"]

    # Copy best model to current output_folder
    shutil.copy(args.resume_train.replace("last_checkpoint.pth", "best_model.pth"), output_folder)

    return model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num

def build_model(args):
    geoLocalizationLayer = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

    args.encoder_dim = 512
    if args.grl:
        grl_discriminator = network.get_discriminator(args.encoder_dim, len(args.grl_datasets.split("+")))
    else:
        grl_discriminator = None

    model = network.AttenNetVLAD(args.backbone, geoLocalizationLayer, grl_discriminator, args.attention)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    return model.to(args.device).train()

def load_datasets(args):
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L, current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Validation set: {val_ds}")
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1", positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Test set: {test_ds}")

    grl_ds = None
    if args.grl:
        grl_ds = GrlDataset(args.grl_datasets.split("+"))


    return groups, classifiers, classifiers_optimizers, val_ds, test_ds, grl_ds