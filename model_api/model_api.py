import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
import os

import torch
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_path = "config.yaml" if "config.yaml" in os.listdir() else "Recommender/config.yaml"

with open(config_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

model = SentenceTransformer(cfg['model']['backbone']['base'], device=device)
model.max_seq_length = cfg['model']['max_sentence_len']

def prompt_to_vec(prompts):
    prompt_emb = model.encode(prompts,
                 batch_size=cfg['model']['batch_size'],
                 device=device,
                 show_progress_bar=True,
                 convert_to_numpy=True,
                 normalize_embeddings=True)

    return prompt_emb

def get_single_embeddings(data:list, weights:list):
    embeds = []
    for i, feature in enumerate(data):
        if len(feature)>1:
            emb = prompt_to_vec([feature]) * weights[i]
            embeds.append(emb)

    embeds = np.stack(embeds, axis=0).sum(axis=0)

    return embeds

def get_batch_embeddings(data:list, weights:list):
    embeds = []
    for i, feature in enumerate(data):
        #if len(feature)>1:
        emb = prompt_to_vec(feature) * weights[i]
        embeds.append(emb)

    embeds = np.stack(embeds, axis=0).sum(axis=0)

    return embeds

class SingleRequestIn(BaseModel):
    title: str
    description: str
    skills: str

class SingleOut(BaseModel):
    emb: str

class BatchRequestIn(BaseModel):
    title: list
    description: list
    skills: list

class BatchOut(BaseModel):
    emb: list

@app.post("/single_embeddings", response_model=SingleOut)
async def get_preds(user_request: SingleRequestIn):
    '''
    :param user_request:  {'title': "Barman",
                            'description': 'The best Barman',
                            'skills': 'drinks'}
    :return: {'emb': '[[-0.033849991858005524, 0.037500470876693726, ... -0.018372448161244392]]'}
    '''

    title = user_request.title
    desc = user_request.description
    skills = user_request.skills

    emb = get_single_embeddings(data=[title, desc, skills], weights=[0.5, 0.5, 0.])
    embeds = json.dumps(emb.tolist())
    return {"emb": embeds}
@app.post("/batch_embeddings", response_model=BatchOut)
async def get_preds(user_request: BatchRequestIn):
    '''

    :param user_request:  {'title': ["Data Stewart", "Barman"],
                           'description': ['The best data man, 'The best Barman'],
                           'skills': ['data analysis', 'drinks']}
    :return:  {'emb': ['[-0.016636449843645096, 0.0004079855279996991, ... -0.007354529574513435]',
                       '[-0.033849991858005524, 0.037500470876693726, ... -0.018372448161244392]']}
    '''

    title = user_request.title
    desc = user_request.description
    skills = user_request.skills

    emb = get_batch_embeddings(data=[title, desc, skills], weights=[0.5, 0.5, 0.])
    embeds = [json.dumps(x.tolist()) for x in emb]
    return {"emb": embeds}

def run_model():
    #if __name__ == "__main__":
    uvicorn.run("model_api:app")

if __name__ == "__main__":
    uvicorn.run("model_api:app", host="0.0.0.0", port=8000)

    # url = 'http://127.0.0.1:8000/predict'
    # myobj = {'img': f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"}
    #myobj = {'title': f"Data Stewart", 'description': 'The best data data data', 'skills': 'data analysis'}
    # x = requests.post(url, json=myobj)
    # np.asarray(json.loads(x.json()['emb']))
    #np.asarray(df.iloc[:10]['emb'].apply(lambda x: json.loads(x)).to_list())


    # url = 'http://127.0.0.1:8000/batch_embeddings'
    # myobj = {'title': ["Data Stewart", "Barman"],
    #          'description': ['The best data data data', 'The best Barman Barman Barman'],
    #          'skills': ['data analysis', 'drinks']}
    # x = requests.post(url, json=myobj)
    # dec = np.asarray([json.loads(X) for X in x.json()['emb']])

    # url = 'http://127.0.0.1:8000/single_embeddings'
    # myobj = {'title': "Barman",
    #          'description': 'The best Barman Barman Barman',
    #          'skills': 'drinks'}
    # x = requests.post(url, json=myobj)
    # dec = np.asarray(json.loads(x.json()['emb']))