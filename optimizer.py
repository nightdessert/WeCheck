import torch.optim as optim
import torch.nn as nn

class Optimizer():
    def __init__(self,args, model,total_step):
        self.args = args
        self.lr = args.lr
        self.warmup = args.warmup_steps
        self.epoch = args.epoch
        self.grad_norm = args.grad_norm
        self.total_step = total_step
        #self.warmup_factor = all_step_cnt/self.warmup
        self.optim = optim.Adam(model.parameters(), weight_decay=0.01, eps=1e-6)
    
    def update_lr(self,step_cnt,model):
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm)
        if step_cnt <= self.warmup:
            warmup_factor = step_cnt / self.warmup
            lr = warmup_factor * self.lr
        else:
            pct_remaining = 1 - (step_cnt - self.warmup) / (self.total_step  - self.warmup)
            lr = self.lr * pct_remaining
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        self.optim.step()
        self.optim.zero_grad()