import glob,random,json,os
import torch
import numpy as np
from torch.utils.data import Dataset
from snorkel.labeling.model import LabelModel

scorer = ['anli_score', 'summac_score', 'qafacteval_score']
def load_dataset(args):
    base_path = args.src_dir
    dataset_name = ['xsum','cnndm','paws','dialogue']
    #dataset_name = ['paws']
    paths = [base_path.format(d,d) for d in dataset_name]
    datasets = []
    for d_name, d in zip(dataset_name,paths):
        print(d_name)
        files = [json.loads(line) for p in glob.glob(d) for line in open(p)]
        datasets.append((d_name, files))
    return datasets

def collate_fn(batch, tok):
    batch_prems = [x[0] for x in batch]
    batch_hypos = [x[1] for x in batch]
    dataset_index = torch.tensor([x[3] for x in batch])
    batch_scores = torch.tensor([x[2] for x in batch],dtype=torch.float)
    batch_tokens = tok.batch_encode_plus(list(zip(batch_prems,batch_hypos)), padding=True, 
            truncation=True, max_length=512, return_tensors="pt", truncation_strategy="only_first")
    results = {
        "inputs": batch_tokens,
        "gold_score": batch_scores,
        "dataset":dataset_index,
    }
    return results

def get_label(score, threshold,dataset):
    label = []
    for i,s in enumerate(score):
        s_thresh = threshold[dataset][scorer[i]]
        if s>s_thresh[1]:
            label.append(1)
        elif s<s_thresh[0]:
            label.append(0)
        else:
            label.append(-1)
    return label

def learn_snorkel(args,datasets,threshold):
    labels = []
    dt = []
    for d in datasets:
        for example in d[1]:
            example_score = []
            examples = [[example[key][i] for key in scorer] for i in range(len(example[scorer[0]]))]
            for s in  examples:
                labels.append(get_label(s,threshold,d[0]))
    label_model = LabelModel(cardinality=2, verbose=False)
    #label_model = LabelModel(m=3)
    label_model.fit(np.array(labels), n_epochs=500, log_freq=50, seed=123)
    #label_model.fit(np.array(labels))
    probs = label_model.predict_proba(np.array(labels))[:,1]
    lowp, highp = np.percentile(probs,args.lowp), np.percentile(probs,args.highp)
    dt.append((lowp,highp))
    dt.append((lowp,highp))
    dt.append((lowp,highp))
    dt.append((lowp,highp))
    return label_model,dt

def average(scores):
    return sum(scores)/len(scores)

def major_vote(scores):
    pvote, nvote = 0,0
    for s in scores:
        if s==1: pvote+=1
        elif s==0:nvote+=1
    if pvote > nvote: return 1
    elif nvote > pvote: return 0
    else: return -1

def naive_predict(scores):
    pvote, nvote = 0,0
    for s in scores:
        if s>0.5: pvote+=1
        elif s<0.5:nvote+=1
    if pvote > nvote:
        return 1
    elif nvote > pvote:
        return 0
    else:
        return average(scores) >0.5
def ave_cof(scores):
    return sum(scores)/len(scores)


def original(scores,other):
    return scores

def snorkel(scores,labeler,threshold):
    score = labeler.predict_proba(np.array(scores))[0]
    if score[1]>=threshold[1] or score[1]<=threshold[0]:
        return score[1]
    else:
        return score[1]

def dataset_threshold(dataset):
    thresh_result = {}
    thresh_dataset = {
        'xsum':(25,75),
        'cnndm':(10,75),
        'paws':(25,75),
        'dialogue':(25,75),
    }
    for d in dataset:
        thresh_result[d[0]] = {}
        for key in scorer:
            thresh = thresh_dataset[d[0]]
            scores = sorted([s for e in d[1] for s in e[key]])
            lowp, highp = np.percentile(scores, thresh[0]),np.percentile(scores,thresh[1])
            thresh_result[d[0]][key] = (lowp, highp)
    return thresh_result

        
class WeakSupervisonData(Dataset):
    def __init__(self,args, tok):
        self.args = args
        self.tok = tok
        self.datasets = load_dataset(args)
        self.scorer_threshold = dataset_threshold(self.datasets)
        print(self.scorer_threshold)
        self.low_thresh, self.high_thresh = [], []
        if args.method=='ave_conf':
            self.agg_fun = ave_cof
        elif args.method=='naive_predict':
            self.agg_fun = naive_predict
        elif args.method=='major_vote':
            self.agg_fun = major_vote
        elif args.method=='snorkel':
            self.labeler, self.thresh = learn_snorkel(args,self.datasets,self.scorer_threshold)
            #self.labeler, self.thresh = sperate_snorkel(self.datasets,self.scorer_threshold)
            print(self.thresh)
            self.agg_fun = snorkel
        elif args.method=='weaseal':
            self.agg_fun = None
        self.datasets = self.preprocess_dataset()
    def __len__(self):
        return len(self.datasets)

    def preprocess_dataset(self):
        datasets, gold = [], []
        for idx, d in enumerate(self.datasets):
            dataset = []
            for example in d[1]:
                hype =  example['beams']
                premise =  [example['article']]*len(hype)
                examples = [[example[key][i] for key in scorer] for i in range(len(hype))]
                if self.args.method=='ave_conf':
                    scores = [self.agg_fun(s) for s in examples] 
                    beam_data = [[p, h ,s_k,idx] for p,h,s_k in zip(premise,hype, scores)]                    
                elif self.args.method=='major_vote':
                    sk_scores = [self.agg_fun(get_label(s, self.scorer_threshold,d[0])) 
                        for s in examples]
                    beam_data = [[p, h ,s_k,idx] for p,h,s_k in zip(premise,hype, sk_scores) if s_k>-1]
                elif self.args.method=='snorkel':
                    threshold = self.thresh[idx]

                    sk_scores = [self.agg_fun([get_label(s, self.scorer_threshold,d[0])],self.labeler,threshold) 
                        for s in examples]
                    beam_data = [[p, h ,s_k,idx] for p,h,s_k in zip(premise,hype, sk_scores) if s_k>-1]
                elif self.args.method=='weaseal':
                    sk_scores = [get_label(s, self.scorer_threshold,d[0])
                        for s in examples]
                    beam_data = [[p, h ,s_k,idx] for p,h,s_k in zip(premise,hype, sk_scores)]
                    
                dataset += beam_data
            datasets += dataset
        random.shuffle(datasets)
        print(len(datasets))
        return datasets

    def __getitem__(self, idx):
        return self.datasets[idx]
                
