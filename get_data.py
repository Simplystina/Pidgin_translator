import string
import pandas as pd
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

df = pd.read_excel(r"PD Data.xlsx")
eng =list(df['English sentences'])
pid =list(df['Pidgin sentences'])
eng_data= []
pid_data = []

for i in range(len(df)):
  if isinstance(pid[i],(str,)):
      for k in eng[i].split("\n"):
        eng_data.append(re.sub(r"[^a-zA-Z0-9]+", ' ', k))
      for l in pid[i].split("\n"):
        pid_data.append(re.sub(r"[^a-zA-Z0-9]+", ' ', l))

data = []
for i,(k,j) in enumerate(zip(eng_data,pid_data)):
  d = {}
  d[i] = i
  d["translate"] = {}
  d["translate"]["en"] =k
  d["translate"]['pd'] = j
  data.append(d)


parallel_sentence = {}
for i in range(len(data)):
    eng = data[i]['translate']['en']
    parallel_sentence[eng] = data[i]['translate']['pd']
        
sent_embed = {}
for i in range(len(data)):
    eg = data[i]['translate']['en']
    sentence_embeddings = model.encode(eg)
    sent_embed[eg] = sentence_embeddings

with open('Parralel_data.pickle', 'wb') as fp:
    pickle.dump(parallel_sentence, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('sentence_embeddings.pickle', 'wb') as fp:
    pickle.dump(sent_embed, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
np.save("data", np.array(data))