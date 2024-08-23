import torch.nn as nn
import torch
from torchvision.transforms.functional import affine, InterpolationMode
import random

class CeLoss(object):
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)  # , weight=tbd)
        self.loss_weight = loss_weight

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=logits, target=target)  # (Num_nodes, num_classes)
        return self.loss_weight * loss


class ClusterPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """
    
    def __init__(self, loss_weight, num_classes=2, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        positives = class_specific_min_distances * target_one_hot  # shape (N, classes)

        if self.reduction == "mean":
            loss = positives.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = positives.sum()

        return self.loss_weight * loss


class SeparationPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=2, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        negatives = class_specific_min_distances * (1 - target_one_hot)  # shape (N, classes)

        if self.reduction == "mean":
            loss = -negatives.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = -negatives.sum()

        return self.loss_weight * loss


class ClusterRoiFeat(object):
    """
    Cluster cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=2, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: cluster loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        # reshaping similarities to group based on class they belong to. shape (N, classes, P_pre_classes)
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        class_specific_max_similarity, _ = similarities.max(dim=2)  # Shape = (N, classes)
        # pick similarity scores of classes the input belongs to
        positives = class_specific_max_similarity * target_one_hot  # shape (N, classes)
        loss = -1 * positives  # loss is negative sign of similarity scores

        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class SeparationRoiFeat(object):
    """
    Separation cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=2, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: separation loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        # reshaping similarities to group based on class they belong to. shape (N, classes, P_pre_classes)
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        class_specific_max_similarity, _ = similarities.max(dim=2)  # Shape = (N, classes)
        # pick similarity scores of classes the input belongs to
        negatives = class_specific_max_similarity * (1 - target_one_hot)  # shape (N, classes)
        loss = negatives

        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class OrthogonalityLoss(object):
    """
    orthogonality loss to encourage diversity in learned prototype vectors
    """

    def __init__(self, loss_weight, num_classes=2, mode="per_class"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.mode = mode  # one of 'per_class' or 'all'
        if mode == "per_class":
            self.cosine_similarity = nn.CosineSimilarity(dim=3)
        elif mode == "all":
            self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def compute(self, prototype_vectors):
        """
        compute loss given the prototype_vectors
        :param prototype_vectors: shape (P, channel, 1, 1). P=num_classes x numPrototypes
        :return: orthogonality loss either across each class, summed (or averaged), or across all classes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=prototype_vectors.device)

        # per class diversity
        if self.mode == "per_class":
            # reshape to (num_classes, num_prot_per_class, channel):
            prototype_vectors = prototype_vectors.reshape(self.num_classes, -1, prototype_vectors.shape[1])
            # shape of similarity matrix is (num_classes, num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(prototype_vectors.unsqueeze(1), prototype_vectors.unsqueeze(2))
        elif self.mode == "all":
            # shape of similarity matrix is (num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(
                prototype_vectors.squeeze().unsqueeze(1),
                prototype_vectors.squeeze().unsqueeze(0),
            )
        # use upper traingle elements of similarity matrix (excluding main diagonal)
        loss = torch.triu(sim_matrix, diagonal=1).sum()

        return self.loss_weight * loss


class L_norm(object):
    def __init__(self, mask=None, p=1, loss_weight=1e-4, reduction="sum"):
        self.mask = mask  # mask determines which elements of tensor to be used for Lnorm calculations
        self.p = p
        self.loss_weight = loss_weight
        self.reduction = reduction

    def compute(self, tensor, dim=None):
        if self.loss_weight == 0:
            return torch.tensor(0, device=tensor.device)

        if self.mask != None:
            loss = (self.mask.to(tensor.device) * tensor).norm(p=self.p, dim=dim)
        else:
            loss = tensor.norm(p=self.p, dim=dim)
        if self.reduction == "mean":
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


def get_affine_config():
    config = {
        "angle": random.uniform(-20, 20),
        "translate": (
            0,
            0,
        ),
        "scale": random.uniform(0.6, 1.5),
        "shear": 0.0,
        "fill": 0,
        "interpolation": InterpolationMode.BILINEAR,
    }
    return config


class TransformLoss(object):
    """
    the loss applied on generated ROIs!
    """

    def __init__(self, loss_weight=1e-4, reduction="sum"):
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction="sum")
        self.reduction = reduction

    def compute(self, x, occurrence_map, model, characteristic_index):
        if self.loss_weight == 0:
            return torch.tensor(0, device=x.device)

        # get the affine transform randomly sampled configuration
        config = get_affine_config()

        # transform input and get its new occurrence map
        transformed_x = affine(x, **config)  # shape (N, D, H, W)
        
        occurrence_map_transformed = model.compute_occurence_map(transformed_x, characteristic_index).squeeze(2)  # shape (N, P, H, W)

        # transform initial occurence map
        occurrence_map = occurrence_map.squeeze(2)  # shape (N, P, H, W)
        
        transformed_occurrence_map = affine(occurrence_map, **config)  # shape (N, P, H, W)

        # compute L1 loss
        loss = self.criterion(occurrence_map_transformed, transformed_occurrence_map)
        if self.reduction == "mean":
            loss = loss / (occurrence_map_transformed.shape[0] * occurrence_map_transformed.shape[1])

        return self.loss_weight * loss