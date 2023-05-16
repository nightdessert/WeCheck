# WeCheck
Open-Source code for ACL 2023 paper:
*[WeCheck: Strong Factual Consistency Checker via Weakly Supervised Learning
](https://arxiv.org/abs/2212.10057)*

## Model description
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
## Reporduction
