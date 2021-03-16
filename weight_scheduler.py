import math

def f(x,alpha):
    return (math.exp(x*alpha)-1)/(math.exp(1*alpha)-1)

class WarmupWeight():
    
    WARMUP_STYLES = ['linear', 'exponential', 'constant']

    
    def __init__(self, start_lr, warmup_iter, num_iters, warmup_style=None, last_iter=-1,alpha=1):
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.warmup_style = warmup_style.lower() if isinstance(warmup_style, str) else None
        self.alpha=alpha
        print('warmup style', warmup_style)
        
    def get_lr(self):
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            if self.warmup_style==self.WARMUP_STYLES[0]:
                return float(self.start_lr) * self.num_iters / self.warmup_iter
            elif self.warmup_style==self.WARMUP_STYLES[1]:
                return float(self.start_lr) * f(self.num_iters / self.warmup_iter, self.alpha)
            else:
                return self.start_lr
        else:
            return self.start_lr
    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        return new_lr