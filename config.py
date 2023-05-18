import os
def exp_v1_settings(args):
    args.src_dir = args.src_dir + '/{}/qafacteval_score_*.json' 
    args.pretrained = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    args.epoch = getattr(args, 'epoch', 3)
    args.accumulate_step = getattr(args, "accumulate_step", 1)
    args.warmup_steps = getattr(args, "warmup_steps",100)
    args.grad_norm = getattr(args, "grad_norm", 1.0) # gradient norm
    args.seed = getattr(args, "seed", 970903) # random seed
    args.lr = getattr(args, "lr", 1e-6) # random seed
    args.eval_interval = getattr(args, "eval_interval", 500)
    args.report_freq = getattr(args, "report_freq", 100)
    args.n_tasks = getattr(args, "n_task", 2)
    args.lowp = getattr(args, "lowp", 26)  
    args.highp = getattr(args, "highp", 70)

