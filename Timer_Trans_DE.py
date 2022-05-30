import pandas as pd
import torch
import numpy as np
from transformers import pipeline
import time
from transformers import MarianTokenizer, MarianMTModel
from typing import List

print("Enter CSV_FILE_NAME")
path=input()

ds1=pd.read_csv(path)

src = 'en'  # source language
trg = 'de'  # target language
mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
model = MarianMTModel.from_pretrained(mname)
tok = MarianTokenizer.from_pretrained(mname)
def en_de(txt):
  batch = tok.prepare_seq2seq_batch(src_texts=[txt])  # don't need tgt_text for inference
  gen = model.generate(**batch)  # for forward pass: model(**batch)
  words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)  # returns "Where is the bus stop ?"
  return(words[0])

model_name = f'Helsinki-NLP/opus-mt-{trg}-{src}'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model1 = MarianMTModel.from_pretrained(model_name)
def de_en(txtx):
  translated = model1.generate(**tokenizer.prepare_seq2seq_batch([txtx]))
  tgt = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
  return(tgt)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

time_tran=[]
time_cls=[]


list1=ds1.perturbed_text
list2=[]
a=0

for x in list1[:20]:
  cnta=time.time()
  w1 = en_de(x)
  list2.append(de_en(w1))
  cntb=time.time()
  time_tran.append(cntb-cnta)
  print(f"Example number {a}")
  a+=1


translated=list2
perturbed=list1
initial_prob=list(ds1.original_output)
initial_score=list(ds1.original_score)
perturbed_prob=list(ds1.perturbed_output)
perturbed_score=list(ds1.perturbed_score)

final_prob=[]
final_score=[]



a=0
for y in translated:
  cnta= time.timeit()
  txf1=tokenizer.encode_plus(y,return_tensors="pt",max_length=512)
  txf2=model(**txf1)[0]
  results=torch.softmax(txf2, dim=1).tolist()[0]
  final_prob.append(np.argmax(results))
  final_score.append(results[np.argmax(results)])
  cntb=time.timeit()
  time_tran[a]=time_sum[a]+(cntb-cnta)
  print(f"Iteration number={a}")
  a+=1

a=0
for y in perturbed[:20]:
  cnta= time.timeit()
  txf1=tokenizer.encode_plus(y,return_tensors="pt",max_length=512)
  txf2=model(**txf1)[0]
  results=torch.softmax(txf2, dim=1).tolist()[0]
  final_prob.append(np.argmax(results))
  final_score.append(results[np.argmax(results)])
  cntb=time.timeit()
  time_cls.append(cntb-cnta)
  print(f"Iteration number={a}")
  a+=1


print(f"Average time w/o defense={np.mean(time_cls)}")
print(f"Average time w defense_German={np.mean(time_tran)}")
