import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import pickle
import flask
from flask import Flask, jsonify, request
import sentence_transformers
cwd = os.getcwd()
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

file = f"{cwd}\Parralel_data.p"
with open(file, 'rb') as handle:
    Parallel_data = pickle.load(handle)

with open(f"{cwd}\sentence_embeddings.p", 'rb') as handle:
    sentence_embeddings = pickle.load(handle)

data  = np.load(cwd+r"\data.npy",allow_pickle=True)
data =list(data)

def translate_sentence(sentence):
    if sentence == "":
        return
    result = ""
    example_embed = model.encode(sentence)
    max_similarity = 0
    max_word = ""
    for i in range(len(sentence_embeddings)):
        eg = data[i]['translate']['en']
        result=util.pytorch_cos_sim(example_embed, sentence_embeddings[eg])
        ans = result.numpy().flatten()[0] *100
        if ((ans >= 80) and (ans>max_similarity)):
            max_similarity = ans
            max_word = eg
      
    if (max_similarity == 0):
        result = "Model isn't familiar with this sentence!"
    else:
        result = Parallel_data.get(max_word)
    print(result)
    return result

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict_sentence():
    
    eg_sentence = request.args.get("sentence","")
    predictions = translate_sentence(eg_sentence)
    
    # Return on a JSON format
    if predictions is not None:
        result = {"prediction":predictions}
        return jsonify(result),200
    else:
        return "Error making predictions",400

@app.route('/', methods=['GET'])
def index():
    return '<h1>Welcome! API for translating English sentence to Pidgin sentence<h1>', 200
 
if __name__ == '__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=False, host='0.0.0.0',port=port)# set debug to false before deployment