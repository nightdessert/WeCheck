import torch
import sys, os, random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
from functools import partial

from transformers import BartForConditionalGeneration, BartTokenizer

def collate_mp(batch, tok, is_test=False):
  
    article = [x['src_text'] for x in batch]
    target = [x['target_text'] for x in batch]
    src_input_ids = tok.batch_encode_plus(article, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
    result = {
        "src_text": article,
        "target_text": target,
        "src_input_ids": src_input_ids,
    }

    return result

class GenData(Dataset):
    def __init__(self, fdir, model_type):
        source = os.path.join(fdir, model_type + '.source')
        target = os.path.join(fdir, model_type + '.target')
        self.source = [line.strip() for line in open(source)] 
        self.target = [line.strip() for line in open(target)]
        self.num = len(self.source)
        self.total_len = 1024
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        article = self.source[idx]
        target =  self.target[idx]
        result = {
            "src_text": article,
            "target_text": target,
        }
        return result

def run(rank, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if args.dataset=="xsum":
        mname = "facebook/bart-large-xsum"
    else:
        mname = "facebook/bart-large-cnn"
    tok =  BartTokenizer.from_pretrained(mname, verbose=False)
    collate_fn = partial(collate_mp, tok=tok)
    train_set = GenData(f"{args.src_dir}/", args.datatype)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    if args.dataset == "cnndm":
        max_length, min_length, length_penalty = 140, 55, 2.0
    elif args.dataset == "xsum":
        max_length, min_length, length_penalty = 60, 10, 1.0

    model = BartForConditionalGeneration.from_pretrained(mname)
    if is_mp:
            # Using DDP
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
    else:
        model = model.cuda()
    model.eval()
    target_file = open(os.path.join(args.tgt_dir,'result.{}.json').format(gpuid),'w',encoding='utf-8')
    for (i, batch) in enumerate(dataloader):
        with torch.no_grad():
            #print(model)
            summaries = model.module.generate(
                input_ids=batch["src_input_ids"]["input_ids"].to(gpuid),
                attention_mask=batch["src_input_ids"]["attention_mask"].to(gpuid),
                num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                length_penalty=length_penalty,
                early_stopping=True,
            )
            results = []
            dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            for id_batch,(s,t) in enumerate(zip(batch['src_text'],batch['target_text'])):
                start_id = id_batch*16
                results += [{
                    'article':s,
                    'reference':t,
                    'hypophsis':b
                } for b in dec[start_id:start_id + 16]] 
            for r in results:
                target_file.write(json.dumps(r) + '\n')



def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("--src_dir", type=str, help="source file")
    parser.add_argument("--tgt_dir", type=str, help="target file")
    parser.add_argument("-p", "--port", type=int, default=12345, help="port")
    parser.add_argument("--dataset", type=str, default="cnndm", help="dataset")
    parser.add_argument("--datatype", type=str, default="train", help="dataset")
    parser.add_argument("--seed", type=int, default="970903")
    parser.add_argument("--batch_size", type=int, default="8")
    args.port +=  random.randint(0, 100)
    print(args.port)
    args = parser.parse_args()
    main(args)
    
