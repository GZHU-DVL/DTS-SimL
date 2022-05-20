import torch
from torch.nn import functional as F

from .utils import get_pair_indices, reduce_tensor
# for type hint
from typing import Optional
from torch import Tensor

from ..types import SimilarityType, DistanceLossType


def toone(a):
    a1 = a.max(1)[1]
    a1 = a1.view(-1, 1)
    tar = torch.zeros((len(a1), 100), device=a.device).scatter_(1, a1, 1)
    return tar


def sharpen(x: Tensor, temperature: float) -> Tensor:
    sharpened_x = x ** (1 / temperature)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


def softmax_cross_entropy_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = 'none') -> Tensor:
    """
    :param logits: (labeled_batch_size, num_classes) model output of the labeled data
    :param targets: (labeled_batch_size, num_classes) labels distribution for the data
    :param dim: the dimension or dimensions to reduce
    :param reduction: choose from 'mean', 'sum', and 'none'
    :return:
    """
    loss = -torch.sum(F.log_softmax(logits, dim=dim) * targets, dim=dim)

    return loss


def tosim(x):
    ctu = (torch.zeros(x.shape[1], x.shape[1])).cuda()
    for i in range(x.shape[1]):
        ctu[i][i] = 1
    sim = (torch.zeros(x.shape[0], x.shape[1])).cuda()

    for i in range(sim.shape[0]):
        for i1 in range(sim.shape[1]):
            sim[i][i1] = torch.cosine_similarity(x[i], ctu[i1], dim=-1)
    # sim = sim /torch.sum(sim, dim=1, keepdim=True)
    return sim


def toonehot(x):
    weizhi = x.max(dim=1)[1]
    zero = (torch.zeros(x.shape[0], x.shape[1])).cuda()
    for i, i1 in zip(weizhi, zero):
        i1[i] = 1
    return zero


class PairLoss:
    def __init__(self,
                 similarity_metric: SimilarityType,
                 distance_loss_metric: DistanceLossType,
                 confidence_threshold: float,
                 similarity_threshold: float,
                 similarity_type: str,
                 distance_loss_type: str,
                 reduction: str = "mean"):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

        self.similarity_type = similarity_type
        self.distance_loss_type = distance_loss_type

        self.reduction = reduction

        self.similarity_metric = similarity_metric
        self.distance_loss_metric = distance_loss_metric

    def __call__(self,
                 logits: Tensor,
                 probs: Tensor,
                 targets: Tensor,
                 *args,
                 indices: Optional[Tensor] = None,
                 **kwargs) -> Tensor:
        """

        Args:
            logits: (batch_size, num_classes) predictions of batch data
            probs: (batch_size, num_classes) softmax probs logits
            targets: (batch_size, num_classes) one-hot labels

        Returns: Pair loss value as a Tensor.

        """
        if indices is None:
            indices = get_pair_indices(targets, ordered_pair=True)
        total_size = len(indices) // 2

        i_indices, j_indices = indices[:, 0], indices[:, 1]
        targets_max_prob = targets.max(dim=1).values

        return self.compute_loss(logits_j=logits[j_indices],
                                 probs_j=probs[j_indices],
                                 targets_i=targets[i_indices],
                                 targets_j=targets[j_indices],
                                 targets_i_max_prob=targets_max_prob[i_indices],
                                 total_size=total_size)

    def compute_loss(self,
                     logits_j: Tensor,
                     probs_j: Tensor,
                     targets_i: Tensor,
                     targets_j: Tensor,
                     targets_i_max_prob: Tensor,
                     total_size: int):
        # conf_mask should not track gradient
        ti = sharpen(targets_i, 0.5)
        conf_mask = (ti.max(1)[0] > self.confidence_threshold).detach().float()
        # sime = tosim(targets_i)
        one = toone(targets_i)

        similarities: Tensor = self.get_similarity(targets_i=targets_i,
                                                   targets_j=targets_j,
                                                   dim=1)
        # sim_mask should not track gradient
        sim_mask = F.threshold(similarities, self.similarity_threshold, 0).detach()

        distance = softmax_cross_entropy_loss(logits_j, one, dim=1, reduction="none")
        loss = conf_mask * sim_mask * distance

        if self.reduction == "mean":
            loss = torch.sum(loss) / total_size
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

    def get_similarity(self,
                       targets_i: Tensor,
                       targets_j: Tensor,
                       *args,
                       **kwargs) -> Tensor:
        x, y = targets_i, targets_j

        return self.similarity_metric(x, y, *args, **kwargs)

    def get_distance_loss(self,
                          logits: Tensor,
                          probs: Tensor,
                          targets: Tensor,
                          *args,
                          **kwargs) -> Tensor:
        if self.distance_loss_type == "prob":
            x, y = probs, targets
        else:
            x, y = logits, targets

        return self.distance_loss_metric(x, y, *args, **kwargs)
