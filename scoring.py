import json
import glob
import torch, os,sys
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import numpy as np
import random
import tqdm,transformers
import pandas as pd

import torch.nn as nn
from random import sample
from torch.utils.data import Dataset, DataLoader
from summac.model_summac import SummaCZS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time

logging.disable(logging.WARNING)
def collate_fn(batch):
    return batch[0]

class Data(Dataset):
    def __init__(self,args):
        self.args = args
        if args.scorer=='anli':
            self.paths = glob.glob(args.src_dir + '/?.json')[:2]
            self.tok = AutoTokenizer.from_pretrained('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
        elif args.scorer=='summac':
            self.paths = glob.glob(args.src_dir + '/anli*.json')
        elif args.scorer=='qafacteval':
            self.paths = glob.glob(args.src_dir + '/summac*.json')
        dataset = [line  for p in self.paths for line in open(p)]
        self.datasets = self.flatten_data(dataset)

    def flatten_data(self,datasets):
        results = []
        for line in datasets:
            line = json.loads(line,encoding='utf-8')
            results.append(line)
        return results

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        line = self.datasets[idx]
        if self.args.scorer=='anli' or self.args.scorer=='wecheck':
            h = line['beams']
            batch_prems = [line['article']]*len(h)
            batch_tokens = self.tok.batch_encode_plus(list(zip(batch_prems,h)), padding=True,\
                truncation=True, max_length=1024, return_tensors="pt", truncation_strategy="only_first")
            line['batch_tokens']=batch_tokens
        elif self.args.scorer=='summac':
            line['summac_score'] = []
        elif self.args.scorer=='qafacteval':
            line['qafacteval_score'] = []
        
        return line

def split_batch(batch, batch_size):
    batch1 = {k:v[:batch_size,:] for k, v in batch.items()}
    batch2 = {k:v[batch_size:,:] for k, v in batch.items()}
    return [batch1, batch2]

def run(rank, args):
    #transformers.logging.set_verbosity_error()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    dataset = Data(args)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,            
            collate_fn=collate_fn,sampler=train_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    if args.scorer=='anli' or args.scorer =='wecheck':
        model = AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    if is_mp:
            # Using DDP
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if args.scorer=='anli'or args.scorer =='wecheck':
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
            model.eval()
        elif args.scorer=='summac':
            transformers.logging.set_verbosity_error()
            model = SummaCZS(granularity="sentence", model_name="vitc",device=gpuid).to(gpuid)
            model.eval()
        elif args.scorer=='qafacteval':
            os.environ['TMPDIR']='./cache'
            sys.path.insert(0, './QAFactEval')
            from QAFactEval.qafacteval import QAFactEval
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuid)
            #print(gpuid) 
            model_folder = "./QAFactEval/models"
            kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
                "verbose": False, "generation_batch_size": 128, \
                "answering_batch_size": 128, "lerc_batch_size": 12}
            model = QAFactEval(
                lerc_quip_path=f"{model_folder}/quip-512-mocha",
                generation_model_path=f"{model_folder}/generation/model.tar.gz",
                answering_model_dir=f"{model_folder}/answering",
                lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
                lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
                **kwargs
            )
        
    else:
        model = model.cuda()
   
    results = []
    source, hypo, batches = [], [] , []
    if is_master:
        iterator = tqdm.tqdm(enumerate(dataloader),total=len(dataloader))
    else:
        iterator = enumerate(dataloader)
    input_len = []
    for i,batch in iterator:
        outputs = []
        with torch.no_grad():
            if args.scorer=='anli':    
                batch_tokens = batch['batch_tokens']
                output = model(batch_tokens["input_ids"])
                output = torch.softmax(output["logits"], -1).tolist()
                outputs += [s[0] for s in output]
                batch['anli_score'] = outputs
                del batch['batch_tokens']
                results.append(batch)
            elif args.scorer=='wecheck':    
                batch_tokens = batch['batch_tokens']
                output = model(batch_tokens["input_ids"])
                output = torch.sigmoid(output['logits'])[:,0].tolist() 
                outputs += output
                batch['wecheck_score'] = outputs
                del batch['batch_tokens']
                results.append(batch)
            elif args.scorer=='summac':  
                source =  [batch['article']]*len(batch['beams'])
                outputs += [model.score([src], [gen])["scores"][0] for gen, src in zip(batch['beams'], source)]
                batch['summac_score'] = outputs
                results.append(batch)
            elif args.scorer=='qafacteval':
                source  += [batch['article']]*len(batch['beams'])
                hypo   += [[l] for l in batch['beams']]
                batches.append(batch)
                if len(batches)==200 or (i==len(dataloader)-1 and len(batches)>0):
                    score = model.score_batch_qafacteval(source, hypo,return_qa_pairs=True)
                    score = [min(s[0]['qa-eval']['lerc_quip']/5,1) for s in score]
                    for bi, b in enumerate(batches):
                        b['qafacteval_score'] = score[bi:bi+16]
                        results.append(b)
                    source, hypo, batches = [], [] , []
   
    with open(args.tgt_dir + '/{}_score_{}.json'.format(args.scorer,rank),'w') as target:
        for line in results:
            target.write(json.dumps(line)+'\n')

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
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--seed", type=int, default="970903")
    parser.add_argument("--batch_size", type=int, default="8")
    parser.add_argument("--scorer", type=str, default="anli")
    
    args = parser.parse_args()
    args.port +=  random.randint(0, 100)
    main(args)
        

