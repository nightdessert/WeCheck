import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import os
import random
from utils import Recorder
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from optimizer import Optimizer
import logging
from config import exp_v1_settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import metrics
from dataloader import WeakSupervisonData, collate_fn
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


def cal_score(can,src, tokenizer, model,guid):
    batch_size = 16
    outputs = []
    for i in range(0,len(src),batch_size):
        batch_hypos, batch_prems = can[i:i+batch_size], src[i:i+batch_size]
        batch_tokens = tokenizer.batch_encode_plus(list(zip(batch_prems,batch_hypos)),   
            padding=True, truncation=True, max_length=512, return_tensors="pt", truncation_strategy="only_first")
        batch_tokens = {k: v.to(guid) for k, v in batch_tokens.items()}
        with torch.no_grad():
            output = model(batch_tokens["input_ids"].cuda())
            output = torch.sigmoid(output['logits'][:,0]).tolist()
            outputs += output
    return outputs

def test(model,tok,guid):
    results = {}
    all_result = []
    test_dataset = ['sum','paraphrase','dialogue']

    for task in test_dataset:
        bench_path = "./test_data/{}".format(task)
        model.eval()
        results[task] = []
        for filename in os.listdir(bench_path):
            file_path = os.path.join(bench_path, filename)
            df = pd.read_csv(file_path, encoding='utf8')
            ground = [l for l in df['grounding']]
            generated_text  = [l for l in df['generated_text']]
            score = cal_score(generated_text, ground, tok, model,guid)
            gold = df['label']
            auc = metrics.roc_auc_score(gold, score)
            print(filename, auc)
            results[task].append(auc)
            all_result.append(auc)
        print(task, ":", sum(results[task])/len(results[task]))
    all_acc = sum(all_result)/len(all_result)
    print("average score: ", all_acc)
    return all_acc

def run(rank, args):
    
    if args.config == "exp_v1":
        os.environ['TMPDIR']='./cache'
        exp_v1_settings(args)
    # my parameter settings
    # task initialization
    print("args")
    print(args.lowp, args.highp)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(id, args.log)
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    train_collate_fn = partial(collate_fn, tok=tok)
    train_set = WeakSupervisonData(args, tok)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	    train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, 
            collate_fn=train_collate_fn, sampler=train_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, 
            collate_fn=train_collate_fn)
        
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained)
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)

        else:
            model = model.to(gpuid)
            
    model.train()
    # set the model to scoring mode
    loss_func =  nn.BCELoss()
    check_loss = nn.BCELoss(reduction='none')
    total_step = args.epoch*len(dataloader)
    s_optimizer = Optimizer(args,model, total_step)
    if is_master:
        recorder.write_config(args, [model], __file__)
   
    
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    # start training
    all_step_cnt = 0
    
    print(total_step)
    best_acc = 0
    for epoch in range(args.epoch):
        s_optimizer.optim.zero_grad()
        step_cnt = 0
        epoch_step = 0
        avg_loss,avg_ce_loss = 0, 0
        for (i, batch) in enumerate(dataloader):
            step_cnt += 1
            output = model(batch["inputs"]["input_ids"].to(gpuid))
            prob = torch.sigmoid(output['logits'])[:,0]
            #weak_label = (batch['gold_score'].to(gpuid)>0.5).float()
            weak_label = batch['gold_score'].to(gpuid).float()
            confidence = torch.abs(2*weak_label-1)
            #weak_label = (weak_label>0.5).float()
            ce_loss = check_loss(prob, weak_label) 

           
            ce_loss = torch.mean(ce_loss)
            loss = ce_loss
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            avg_ce_loss += ce_loss.item() / args.accumulate_step
            loss.backward()
            if step_cnt == args.accumulate_step:
                # updating
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                s_optimizer.update_lr(all_step_cnt,model)
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                # report stats
                print("id: %d"%id)
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f,  avg mle loss: %.6f"
                %(epoch+1, epoch_step, avg_loss / args.report_freq, (avg_ce_loss / args.report_freq)**0.5))
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.plot("ce_loss", {"loss": avg_ce_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_ce_loss, avg_loss = 0, 0 
            del loss, ce_loss 
            if all_step_cnt % args.eval_interval == 0  and step_cnt == 0 and  all_step_cnt>500:
                # evaluate the model as a scorer
                if is_master:
                    acc = test(model,tok,gpuid)
                    model.train()
                    if acc> best_acc:
                        best_acc = acc
                        model.module.save_pretrained("model/model_{}.pt".format(acc))

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
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lowp", type=float, default=10)
    parser.add_argument("--highp", type=float, default=10)
    parser.add_argument("--accumulate_step", type=int, default=3)
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")    
    parser.add_argument("--config", default="exp_v1", type=str,help="config path")
    parser.add_argument("--src_dir", default="data/{}/{}_beam/data/qafacteval_score_*.json'", type=str,help="config path")
    parser.add_argument("--method", default="ave_conf", type=str,help="config path")

    args = parser.parse_args()
    args.port +=  random.randint(0, 100)
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
