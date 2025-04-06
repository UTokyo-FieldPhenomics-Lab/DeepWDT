# Build warmup scheduler
import math
from functools import partial

def build_warmup(cfg, base_lr=0.01):
    print('==============================')
    print('WarmUpScheduler: {}'.format(cfg['warmup']))
    print('--base_lr: {}'.format(base_lr))
    print('--warmup_factor: {}'.format(cfg['warmup_factor']))
    print('--wp_iter: {}'.format(cfg['wp_iter']))

    warmup_scheduler = WarmUpScheduler(
        name=cfg['warmup'], 
        base_lr=base_lr, 
        wp_iter=cfg['wp_iter'], 
        warmup_factor=cfg['warmup_factor']
        )
    
    return warmup_scheduler

                           
# Basic Warmup Scheduler
class WarmUpScheduler(object):
    def __init__(self, 
                 name='linear', 
                 base_lr=0.01, 
                 wp_iter=500, 
                 warmup_factor=0.00066667):
        self.name = name
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor


    def set_lr(self, optimizer, lr, base_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / base_lr
            param_group['lr'] = lr * ratio


    def warmup(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        if self.name == 'exp':
            tmp_lr = self.base_lr * pow(iter / self.wp_iter, 4)
            self.set_lr(optimizer, tmp_lr, self.base_lr)

        elif self.name == 'linear':
            alpha = iter / self.wp_iter
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            tmp_lr = self.base_lr * warmup_factor
            self.set_lr(optimizer, tmp_lr, self.base_lr)


    def __call__(self, iter, optimizer):
        self.warmup(iter, optimizer)


# Added cosine annealing

def get_lr_scheduler(lr, warmup_total_iters, no_aug_iters, total_iters, warmup_lr_start = 0, min_lr_ratio = 0.2):

    lr_func = partial(
        yolox_warm_cos_lr,
        lr,
        min_lr_ratio,
        total_iters,
        warmup_total_iters,
        warmup_lr_start,
        no_aug_iters,
            )

    return lr_func
       

def yolox_warm_cos_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    no_aug_iter,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(
            math.pi
            * (iters - warmup_total_iters)
            / (total_iters - warmup_total_iters - no_aug_iter)
            )
        )
    return lr


def set_optimizer_lr(optimizer, lr_scheduler_func, iter):
    lr = lr_scheduler_func(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr