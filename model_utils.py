import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
import torch
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

model = SentenceTransformer(cfg['model']['backbone']['base'], device=device)
model.max_seq_length = cfg['model']['max_sentence_len']


def get_embeddings(prompts):
    prompt_emb = model.encode(prompts,
                 batch_size=cfg['model']['batch_size'],
                 device=device,
                 show_progress_bar=True,
                 convert_to_numpy=True,
                 normalize_embeddings=True)

    return prompt_emb

def get_dataset_embeddings(df, columns: list = ['Category', 'Resume'], weights: list = [0.5, 0.5]):
    assert len(columns) == len(weights)

    embeds = []
    for i, col in enumerate(columns):
        emb = get_embeddings(df[col]) * weights[i]
        embeds.append(emb)

    embeds = np.stack(embeds, axis=0).sum(axis=0)

    return embeds

def get_rec_from_dataset(df_resume, df_jobs, prompt_embeddings, jobs_embeddings, num_rec = 5):

    top_jobs = (-1 * (prompt_embeddings @ jobs_embeddings.T)).argsort(axis=1)[:, :num_rec]
    resume_index = np.random.randint(0, len(df_resume))
    print(df_resume.iloc[resume_index].Resume[:1024])

    return df_jobs.iloc[top_jobs[resume_index]]


def get_rec_from_scratch(data:list, weights:list, df_jobs, jobs_embeddings, num_rec = 5):
    embeds = []
    for i, feature in enumerate(data):
        emb = get_embeddings([feature]) * weights[i]
        embeds.append(emb)

    embeds = np.stack(embeds, axis=0).sum(axis=0)

    top_jobs = (-1 * (embeds @ jobs_embeddings.T)).argsort(axis=1)[:, :num_rec]

    print(f'Max scores: {(embeds @ jobs_embeddings.T)[:,top_jobs]}')

    return df_jobs.iloc[top_jobs[0]]



