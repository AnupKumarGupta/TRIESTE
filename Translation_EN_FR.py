import pandas as pd
import torch
import numpy as np
from transformers import MarianTokenizer, MarianMTModel
from typing import List

print("Enter attack csv file name")
path=input()

ds1=pd.read_csv(path)



list1=ds1.perturbed_text
list2=[]
a=0

## Translation pipeline
src = 'en'  # source language
trg = 'fr'  # target language
mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
model = MarianMTModel.from_pretrained(mname)
tok = MarianTokenizer.from_pretrained(mname)
def en_fr(txt):
  batch = tok.prepare_seq2seq_batch(src_texts=[txt])  # don't need tgt_text for inference
  gen = model.generate(**batch)  # for forward pass: model(**batch)
  words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)  # returns "Where is the bus stop ?"
  return(words[0])

model_name = f'Helsinki-NLP/opus-mt-{trg}-{src}'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model1 = MarianMTModel.from_pretrained(model_name)
def fr_en(txtx):
  translated = model1.generate(**tokenizer.prepare_seq2seq_batch([txtx]))
  tgt = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
  return(tgt)


for x in list1:
  w1 = en_fr(x)
  list2.append(fr_en(w1))
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok_name = input("Enter Tokenizer name (corr. to the classifier model) from Hugging Face")
tokenizer = AutoTokenizer.from_pretrained(tok_name)

model_name = input("Enter classifier model name from Hugging Face")
model = AutoModelForSequenceClassification.from_pretrained("model_name")

a=0
for y in translated:
  
  txf1=tokenizer.encode_plus(y[0],return_tensors="pt",max_length=512)
  txf2=model(**txf1)[0]
  results=torch.softmax(txf2, dim=1).tolist()[0]
  final_prob.append(np.argmax(results))
  final_score.append(results[np.argmax(results)])
  print(f"Iteration number={a}")
  a+=1

df = pd.DataFrame(list(zip(initial_prob,initial_score,perturbed_prob,perturbed_score,final_prob,final_score,translated)), 
               columns =['Initial_P', 'Initial_S','Pert_P','Pert_S','Final_P','Final_S','Translated_DE']) 

s=0
len_ds1=len(list1)
for x in range(len(list1)):
  if(df.loc[x,'Initial_P']==df.loc[x,'Final_P']):
    s+=1

print(f"Percentage of successful attacks is {s/len(list1)*100}")

save_file = input("Enter Result file name")
df.to_csv(save_file)
