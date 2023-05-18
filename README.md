# WeCheck
Open-Source code for ACL 2023 paper:
*[WeCheck: Strong Factual Consistency Checker via Weakly Supervised Learning
](https://arxiv.org/abs/2212.10057)*

## Model Description
WeCheck is a factual consistency metric trained from weakly annotated samples.
This open-sourced WeCheck can be used to check the following three generation tasks:

**Text Summarization/Knowlege grounded dialogue Generation/Paraphrase**

This  open-sourced WeCheck checkpoint is trained based on the following three weak labler:

*[QAFactEval
](https://github.com/salesforce/QAFactEval)* / *[Summarc](https://github.com/tingofurro/summac)* / *[NLI warmup](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)* 

This  open-sourced WeCheck checkpoint is trained based on the warmed up checkpoint from (which we denote as NLI_warmup in our paper):

https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

## Quik Usage/Inference
You can simply apply WeCheck for Factual Consistency evaluation for just two-step !

You can also refer to this huggingface page to use and get the trained model checkpoint:

https://huggingface.co/nightdessert/WeCheck

### Step 1: Install transformers&pytorch

```python
conda/pip install
```
### Step 2:  Load Wechck model and inference
load model checkpoint and tokenizer
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "nightdessert/WeCheck"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
usage example: single sample 
```python
premise = "I first thought that I liked the movie, but upon second thought it was actually disappointing." # Input for Summarization/ Dialogue / Paraphrase
hypothesis = "The movie was not good." # Output for Summarization/ Dialogue / Paraphrase
input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt", truncation_strategy="only_first", max_length=512)
output = model(input["input_ids"].to(device))['logits'][:,0]  # device = "cuda:0" or "cpu"
prediction = torch.sigmoid(output).tolist()
print(prediction) #0.884
```
usage example: a batch of samples
```python
premise = [] # Input sample list for Summarization/ Dialogue / Paraphrase
hypothesis = [] # Output sample list for Summarization/ Dialogue / Paraphrase
batch_size = 8 #  slice the list if you have too many samples
result_scores = []
for i in range(0,len(premise),batch_size):
    batch_tokens = tokenizer.batch_encode_plus(list(zip(premise[i:i+batch_size], hypothesis[i:i+batch_size])), padding=True, 
            truncation=True, max_length=512, return_tensors="pt", truncation_strategy="only_first")
    output = model(batch_tokens["input_ids"].to(device))['logits'][:,0]  # device = "cuda:0" or "cpu"
    prediction = torch.sigmoid(output).tolist()
    result_scores += prediction
print(result_scores)
```
## Reporduction&Source Code
### Major Requirement:
#### Environment:
```text
pytorch #recommand: 1.10.0

transformers # A version that supports AutoModelForSequenceClassification and AutoTokenizer

*[snorkel](https://github.com/snorkel-team/snorkel)* #The pakage for our Labeling function, quick install via 'pip install snorkel
' or 'conda install snorkel -c conda-forge' 
```
### Step 1: Boostrap and Weak  Annotation:
In order to train our metric, we need to first obtain enough data from the target task and annotate them with different weak supervision labelers.

If we have n tasks and m weak supervision labelers, we first boostrap these n task  seperately by taking the  beam search samples,then we save each sample in json format:
```python
{"article":"",  "beams":[beam_1, beam2, ... beam_m]}
```
where "beams" is a list contains all the results from beam seach.

And we annotate the factual consistency of every beam  using  m weak supervision labelers and save all the score in the input by:
```python
{"article":"",  "beams":[beam_1, beam2, ... beam_m], "metric_j_score": [..., beam_i_score, ...]}
```
**If this step is too sophisticated for you, you can directly use our preprocessed data from ./wecheck_data**. 

### Step 1.1: Boostrapping Task Data:
**Summarization:**

Given CNN/DM or XSUM train set locate at $src_dir, where input articles and references are in file x.source, x.reference.

We boostrap from bart using diverse beam search by:
```python
python sum_candidate.py --gpuid 0 1 2 3 --src_dir $src_dir --tgt_dir $tgt_dir --dataset 'cnn/dm' or 'xsum'
```
Then output summries will save in $tgt_dir.

**Dialog:**

For dialog, we use *[parlai](https://parl.ai/)* for boosttrap, we use Mem-
Net and dodecaDialogue trained on Wow dataset.

You can get this data by parlai command and preprocess them into the unified format mentioned above.

**Paraphrase:**

As this task is relatively easy, we directly apply samples from PAWS.

### Step 1.2: Weak  Annotation:

Annotate all the samples boostrapped from generation models and annotate their factulity using the following three weak labeler:

*[QAFactEval
](https://github.com/salesforce/QAFactEval)* / *[Summarc](https://github.com/tingofurro/summac)* / *[NLI warmup](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)* 
We use a unified function 'scoring.py' to annotate different labeler.

**NLI_warmup Annotation:** 
We first annotate $tgt_dir with *NLI_warmup and ouput samples in $nliwarup_dir:
```python
python scoring.py --gpuid 0 1 2 3 --src_dir $tgt_dir --tgt_dir ./wecheck_data/task_name/ --scorer anli
```

**Summac Annotation:** 
Put *[Summarc](https://github.com/tingofurro/summac)* in the path and annotate $nliwarup_dir and ouput samples in $summac_dir:
```python
python scoring.py --gpuid 0 1 2 3 --src_dir ./wecheck_data/task_name/ --tgt_dir ./wecheck_data/task_name/ --scorer summac
```

**QAFactEval Annotation:** 
Put *[QAFactEval](https://github.com/salesforce/QAFactEval)*  in the path and annotate $summac_dir and ouput samples in $qafacteval_score (you may need to independetly the environment of QAFactEval which may be confict with Summac and NLI):
```python
python scoring.py --gpuid 0 1 2 3 --src_dir ./wecheck_data/task_name/ --tgt_dir ./wecheck_data/task_name/ --scorer qafacteval
```
### Step 2: Training WeCheck:
After the most complex preprocessing step, we easily train WeCheck by: 
```python
python main.py --cuda --gpuid 0 1 2 3 --src_dir ./wecheck_data

