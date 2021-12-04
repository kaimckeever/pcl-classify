import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from dont_patronize_me import DontPatronizeMe

dpm = DontPatronizeMe("data/", "dontpatronizeme_pcl.tsv")
dpm.load_task1()
df = dpm.train_task1_df
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")
embeddings = []
encoded_input = tokenizer(
    df["text"][: len(df) / 2].tolist(),
    df["text"][len(df) / 2 :].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
embeddings = model(**encoded_input)
embeddings = embeddings[0].detach().numpy()
df = df.assign(embeddings=embeddings)
df.to_csv("data/task1_bert_embeddings.tsv", sep="\t", index=False)
