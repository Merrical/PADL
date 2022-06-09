def adjust_learning_rate(optimizer, i_iter, LR, EPOCH, POWER):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(LR, i_iter, EPOCH, POWER)
    optimizer.param_groups[0]['lr'] = lr
    # return lr


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)

