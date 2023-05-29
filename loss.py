import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class CEDice(nn.Module):
    def __init__(self, dice_weight=0.2,num_classes=6):
        super(CEDice, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim = 1)
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            for c in range(self.num_classes):
                jaccard_target = (targets == c).float()
                jaccard_output = outputs[:, c].exp()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= ((2 * intersection + eps) / (union + eps) * self.jaccard_weight / (self.num_classes))
        return loss

class CELDice(nn.Module):
    def __init__(self, dice_weight=0.2,num_classes=6):
        super(CELDice, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim = 1)
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1.
            for c in range(self.num_classes):
                jaccard_target = (targets == c).float()
                jaccard_output = outputs[:, c].exp()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (torch.log((2 * intersection + eps) / (union + eps)) * self.jaccard_weight / self.num_classes)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, outputs, targets):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

class FLDice(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, dice_weight = 0.2, num_classes = 6):
        super(FLDice, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        loss = ((self.alpha * (1-pt)**self.gamma * ce_loss).mean()) * (1 - self.jaccard_weight)
        if self.jaccard_weight:
            eps = 1.
            outputs = F.softmax(outputs, dim = 1)
            for c in range(self.num_classes):
                jaccard_target = (targets == c).float()
                jaccard_output = outputs[:, c]
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (torch.log((2 * intersection + eps) / (union + eps)) * self.jaccard_weight / self.num_classes)
        return loss
    
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs, targets, sigma):
        loss = torch.mean(0.5 * (torch.log(sigma) + (outputs - targets)**2 / sigma))

        return loss

class EdgeFLDice(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, dice_weight = 0.2, num_classes = 6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """

        targets_ = targets.float()
        w1 = torch.abs(F.avg_pool3d(targets_, kernel_size=3, stride=1, padding=1) - targets_)
        w2 = torch.abs(F.avg_pool3d(targets_, kernel_size=5, stride=1, padding=2) - targets_)
        w3 = torch.abs(F.avg_pool3d(targets_, kernel_size=7, stride=1, padding=3) - targets_)
        omega = 1 + 0.5 * (w1 + w2 + w3)

        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        loss = ((self.alpha * (1-pt)**self.gamma * ce_loss * omega / (omega + 0.5)).mean()) * (1 - self.jaccard_weight)
        if self.jaccard_weight:
            eps = 1.
            outputs = F.softmax(outputs, dim = 1)
            for c in range(self.num_classes):
                jaccard_target = (targets == c).float()
                jaccard_output = outputs[:, c]
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (torch.log((2 * intersection + eps) / (union + eps)) * self.jaccard_weight / self.num_classes)
        return loss