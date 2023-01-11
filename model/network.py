
import torch
import logging
import torchvision
from torch import nn

import numpy as np
import faiss
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import grl_datasets

from model.layers import Flatten, L2Norm, GeM, GradientReversal
from model.vlad_network import NetVLAD, AttenNetVLAD

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim, input_dim, num_classes=2):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = get_aggregation(features_dim, fc_output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

def get_aggregation(features_dim, fc_output_dim):
    aggregation = nn.Sequential(
        L2Norm(),
        GeM(),
        Flatten(),
        nn.Linear(features_dim, fc_output_dim),
        L2Norm(),
    )
    return aggregation

def get_discriminator(args, input_dim, num_classes=2):
    if args.grl:
        discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    else: 
        discriminator = None
    return discriminator


def get_backbone(args, backbone_name):
    if backbone_name.startswith("resnet18"):
        backbone = torchvision.models.resnet18(pretrained=True)

        features_dim = 512
       
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False

        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    backbone = torch.nn.Sequential(*layers)
    
    return backbone, features_dim

def get_clusters(args, cluster_set, model):
    num_descriptors = 50000
    desc_per_image = 40
    num_images = ceil(num_descriptors / desc_per_image)
    if not "biost" in args.train_q: # TODO set a parameter
        cluster_set = Subset(cluster_set, list(range(cluster_set.db_struct.num_gallery)))
    
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), num_images, replace=True))
    dataloader = DataLoader(dataset=cluster_set,
                            num_workers=args.num_workers, batch_size=args.cache_batch_size, 
                            shuffle=False, sampler=sampler)
    with torch.no_grad():
        model = model.eval().to(args.device)
        logging.debug(f"Extracting {'attentive' if args.attention else ''} descriptors ")
        descriptors = np.zeros(shape=(num_descriptors, args.encoder_dim), dtype=np.float32)
        for iteration, (inputs, indices) in enumerate(tqdm(dataloader, ncols=100), 1):
            inputs = inputs.to(args.device)
            encoder_out = model(inputs, mode="feat")
            l2_out = F.normalize(encoder_out, p=2, dim=1)
            image_descriptors = l2_out.view(l2_out.size(0), args.encoder_dim, -1).permute(0, 2, 1)
            batchix = (iteration - 1) * args.cache_batch_size * desc_per_image
            for ix in range(image_descriptors.size(0)):
                sample = np.random.choice(image_descriptors.size(1), desc_per_image, replace=False)
                startix = batchix + ix * desc_per_image
                descriptors[startix:startix + desc_per_image, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
    niter = 100
    kmeans = faiss.Kmeans(args.encoder_dim, args.num_clusters, niter=niter, verbose=False)
    kmeans.train(descriptors)
    logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
    return kmeans.centroids, descriptors

def build_model(args):
    logging.debug(f"Building {'attentive ' if args.attention else ''}NetVLAD {'with GRL' if args.grl else ''}")
    
    args.encoder_dim = 512
    args.fc_output_dim = 512 # ??

    backbone, features_dim = get_backbone(args, "resnet18")
    aggregation = get_aggregation(features_dim, args.fc_output_dim)

    mod_backbone = nn.Sequential(backbone, aggregation)

    netvlad_layer = NetVLAD(num_clusters = args.num_clusters, dim = args.encoder_dim)
    
    if args.grl:
        grl_discriminator = get_discriminator(args.encoder_dim, len(args.grl_datasets.split("+")))
    else:
        grl_discriminator = None
    
    model = AttenNetVLAD(mod_backbone, netvlad_layer, grl_discriminator, attention=args.attention)
    
    if not args.resume:
        cluster_set = grl_datasets.WholeDataset(args.dataset_root, args.train_g, args.train_q)
        logging.debug(f"Compute clustering and initialize NetVLAD layer based on {cluster_set}")
        centroids, descriptors = get_clusters(args, cluster_set, model)
        model.netvlad_layer.init_params(centroids, descriptors)
    
    return model.to(args.device)
