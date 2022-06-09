from loss_func.loss import DiceLoss
import torch.nn as nn


def get_loss_func(args):
    if args.loss_func == "dice":
        loss_func = DiceLoss()
    elif args.loss_func == "bce":
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = DiceLoss()
    return loss_func