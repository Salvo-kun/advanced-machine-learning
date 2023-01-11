
import torch
import logging
import torchvision
import torch.nn.functional as F
import numpy as np
from torch import nn
from math import ceil

import faiss
import resnet
from model.layers import Flatten, L2Norm, GeM, GradientReversal
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
    }


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = get_aggregation(features_dim, fc_output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

class AttenNetVLAD(nn.Module):
    def __init__(self, backbone, netvlad_layer, grl_discriminator=None, attention=False):
        super(AttenNetVLAD, self).__init__()
        self.backbone = backbone
        self.netvlad_layer = netvlad_layer
        self.grl_discriminator = grl_discriminator
        self.weight_softmax = self.backbone.fc.weight
        self.attention = attention

    def forward(self, input, grl=False, mode="vlad"):
        if grl:
            _, out, _ = self.backbone(input)
            out = self.grl_discriminator(out)
        else:
            if self.attention:
                fc_out, feature_conv, feature_convNBN = self.backbone(input)
                bz, nc, h, w = feature_conv.size()
                feature_conv_view = feature_conv.view(bz, nc, h * w)
                probs, idxs = fc_out.sort(1, True)
                class_idx = idxs[:, 0]
                scores = self.weight_softmax[class_idx].to(input.device)
                cam = torch.bmm(scores.unsqueeze(1), feature_conv_view)
                attention_map = F.softmax(cam.squeeze(1), dim=1)
                attention_map = attention_map.view(attention_map.size(0), 1, h, w)
                attention_features = feature_convNBN * attention_map.expand_as(feature_conv)

                if mode == "feat":
                    out = attention_features
                elif mode == "vlad":
                    out = self.netvlad_layer(attention_features)
            else:
                _, out, _ = self.backbone(input)
                if mode == "vlad":
                    out = self.netvlad_layer(out)
        return out

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()

        # Vlad module
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        # Init vlad params
        clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clsts_assign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clsts_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def __vlad_compute_original__(self, x_flatten,soft_assign, N, D):
        vlad = torch.zeros([N, self.num_clusters, D], dtype=x_flatten.dtype, device=x_flatten.device) #24 64 256
        for D in range(self.num_clusters): # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961
            vlad[:,D:D+1,:] = residual.sum(dim=-1)     #vlas.size = 24 64 256 961
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = self.__vlad_compute_original__(x_flatten, soft_assign, N, D)
        return vlad

def get_aggregation(features_dim, fc_output_dim):
    aggregation = nn.Sequential(
        L2Norm(),
        GeM(),
        Flatten(),
        nn.Linear(features_dim, fc_output_dim),
        L2Norm(),
    )
    return aggregation

def get_backbone(args):
    backbone = resnet.resnet18(pretrain="places")

    for name, child in backbone.named_children():
        if name == "layer3":  # Freeze layers before conv_3
            break
        for params in child.parameters():
            params.requires_grad = False
    logging.debug(f"Train only layer3 and layer4 of the {args.backbone}, freeze the previous ones")

    # layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    # backbone = torch.nn.Sequential(*layers)
    features_dim = CHANNELS_NUM_IN_LAST_CONV[args.backbone]

    return backbone, features_dim

    return backbone

def get_discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator

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