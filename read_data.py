import numpy as np
import pandas as pd
import os
import string
import pickle
import flask
from flask import Flask, jsonify, request
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

with open("sentence_embeddings.pickle", 'rb') as handle:
    sentence_embeddings = pickle.load(handle)

with open("Parralel_data.pickle", 'rb') as handle:
    Parallel_data = pickle.load(handle)

data  = np.load("data.npy",allow_pickle=True)
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
        if ((ans >= 90) and (ans>max_similarity)):
            max_similarity = ans
            max_word = eg
      
    if (max_similarity == 0):
        return "Model isn't familiar with this sentence!"
    else:
        result = Parallel_data.get(max_word)
    last_word = sentence.split()[-1]
    end =last_word.translate(str.maketrans('', '', string.punctuation))
    if (last_word[-1] == "?") and (end =="o"*len(end)):
        return f"{result} {'o'*len(end)}?"

    if (last_word[-1] == "?"):
        return f"{result}?"
    if (end =="o"*len(end)):
        return f"{result} {'o'*len(end)}"
    
    return result
 
app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict_sentence():
    
    sentence = request.args.get("sentence","")
    predictions = translate_sentence(sentence)
    
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