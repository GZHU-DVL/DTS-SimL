import torch
from torch.nn import functional as F

from .utils import reduce_tensor, bha_coeff_log_prob, l2_distance

# for type hint
from torch import Tensor

def tosim(x):
    ctu=(torch.zeros(x.shape[1],x.shape[1])).cuda()
    for i in range(x.shape[1]):
        ctu[i][i]=1
    sim = (torch.zeros(x.shape[0],x.shape[1])).cuda()

    for i in range(sim.shape[0]):
        for i1 in range(sim.shape[1]):
            sim[i][i1] = torch.cosine_similarity(x[i],ctu[i1],dim=-1)
    #sim = sim /torch.sum(sim, dim=1, keepdim=True)
    return sim
def toonehot(x):
    weizhi = x.max(dim = 1)[1]
    zero = (torch.zeros(x.shape[0], x.shape[1])).cuda()
    for i,i1 in zip(weizhi,zero):
        i1[i] = 1
    return zero
def toone (a):
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



def mse_loss(prob: Tensor, targets: Tensor, reduction: str = 'mean', **kwargs) -> Tensor:
    return F.mse_loss(prob, targets, reduction=reduction)


def bha_coeff_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param logits: (batch_size, num_classes) model predictions of the data
    :param targets: (batch_size, num_classes) label prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_probs = F.log_softmax(logits, dim=dim)
    log_targets = torch.log(targets)

    # since BC(P,Q) is maximized when P and Q are the same, we minimize 1 - B(P,Q)
    return 1. - bha_coeff_log_prob(log_probs, log_targets, dim=dim, reduction=reduction)


def l2_dist_loss(probs: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    loss = l2_distance(probs, targets, dim=dim)

    return reduce_tensor(loss, reduction)

class SupervisedLoss:
    T = torch.tensor([0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95]).cuda()

    T1=torch.tensor([0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                      0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95]).cuda()
    n = 0
    def __init__(self, reduction: str = 'mean'):
        self.loss_use_prob = False
        self.loss_fn = softmax_cross_entropy_loss
        self.reduction = reduction

    def __call__(self, logits: Tensor, probs: Tensor, targets: Tensor) -> Tensor:
        loss_input = probs if self.loss_use_prob else logits
        loss = self.loss_fn(loss_input, targets, dim=1)

        tru = ((targets.max(dim=1)[1] == probs.max(dim=1)[1]) * 1).cuda()
        truvalue = (tru * probs.max(dim=1)[0]).cuda()
        truposit = ((tru * (probs.max(dim=1)[1] + 1)) - 1).cuda()
        for i in range(len(truposit)):  # range(len(truposit))
            if truposit[i] > -1:
                if SupervisedLoss.T[truposit[i]] > truvalue[i]:
                    SupervisedLoss.T[truposit[i]] = truvalue[i]
        if SupervisedLoss.n<156:
            SupervisedLoss.n+=1
            for i in range(len(truposit)):  # range(len(truposit))
                if truposit[i] > -1:
                    if SupervisedLoss.T1[truposit[i]] > truvalue[i]:
                        SupervisedLoss.T1[truposit[i]] = truvalue[i]
        else:
            SupervisedLoss.T = SupervisedLoss.T1
            SupervisedLoss.n = 0
            SupervisedLoss.T1 = torch.tensor([0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,
                                            0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95]).cuda()
        return reduce_tensor(loss, reduction="mean")


class UnsupervisedLoss:
    
    def __init__(self,
                 loss_type: str,
                 loss_thresholded: bool = False,
                 confidence_threshold: float = 0.,
                 reduction: str = "mean"):
        if loss_type in ["entropy", "cross entropy"]:
            self.loss_use_prob = False
            self.loss_fn = softmax_cross_entropy_loss
        else:
            self.loss_use_prob = True
            self.loss_fn = mse_loss

        self.loss_thresholded = loss_thresholded
        self.confidence_threshold = confidence_threshold
        self.reduction = reduction

    def __call__(self, logits: Tensor, probs: Tensor, targets: Tensor) -> Tensor:
        global loss
        loss_input0 = logits
        loss_input1 = probs

        #sime=tosim(targets)
        t = sharpen(targets, 0.5)
        one = toone(targets)

        loss0 = softmax_cross_entropy_loss(loss_input0, one, dim=1, reduction="none")
        loss1 = self.loss_fn(loss_input1, t, dim=1, reduction="none")

        if self.loss_thresholded:
            
            
            targets_mask0 =(t.max(dim = 1)[0] >= 0.95)*1 # (targets.max(dim=1).values > self.confidence_threshold)
            if len(loss0.shape) > 1:
                # mse_loss returns a matrix, need to reshape mask
                targets_mask0 = targets_mask0.view(-1, 1)
            loss0 *= targets_mask0.float()
            loss0=reduce_tensor(loss0, reduction=self.reduction)

            c = loss_input1.max(dim=1)[1]
            lenth = len(targets)
            M = torch.tensor([0] * lenth).cuda()
            for pos, a0, i1 in zip(c, targets, range(lenth)):
                x = (a0.max(-1)[0] > SupervisedLoss.T[pos])
                M[i1] = x
            M=M*1
            M1=(t.max(dim=1)[0] < 0.95)*1
            targets_mask1 = M * M1  # (targets.max(dim=1).values > self.confidence_threshold)
            if len(loss1.shape) > 1:
                # mse_loss returns a matrix, need to reshape mask
                targets_mask1 = targets_mask1.view(-1, 1)
            loss1 *= targets_mask1.float()
            loss1=reduce_tensor(loss1, reduction=self.reduction)

            loss = loss0  + 150*loss1
           
        return  loss


