import json
from gliner.model import GLiNER

import torch
from tqdm import tqdm
import spacy
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
import os
from types import SimpleNamespace
from gliner.modules.rel_rep import RelMarkerv0


# read jsonl file from data
data = []
with open('../data/docred_expanded.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))


model = GLiNER.from_pretrained("urchade/gliner_small")

model.span_rep_layer = RelMarkerv0(model.config.hidden_size, max_width=12, dropout=0.4)


# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=1000, # number of training iteration
    train_batch_size=2, 
    eval_every=100, # evaluation/saving steps
    save_directory="logs", # where to save checkpoints
    warmup_ratio=0.1, # warmup steps
    device='cpu',
    lr_encoder=1e-5, # learning rate for the backbone
    lr_others=5e-5, # learning rate for other parameters
    freeze_token_rep=False, # freeze of not the backbone
    
    # Parameters for set_sampling_params
    max_types=25, # maximum number of entity types during training
    shuffle_types=True, # if shuffle or not entity types
    random_drop=True, # randomly drop entity types
    max_neg_type_ratio=1, # ratio of positive/negative types, 1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
    max_len=384 # maximum sentence length
)


# Set sampling parameters from config
model.set_sampling_params(
    max_types=config.max_types, 
    shuffle_types=config.shuffle_types, 
    random_drop=config.random_drop, 
    max_neg_type_ratio=config.max_neg_type_ratio, 
    max_len=config.max_len
)


## Predict

text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

ner = []
for ent in doc.ents:
    ner.append([ent.start, ent.end, ent.label_, ent.text])


labels = [
    "is parent of",
    "is capital of",
    "founded by",
    "works for",
    "located in",
    "causes",
    "is a type of",
    "inhibits",
    "discovered by",
    "owns",
    "invests in"
]

tokens = [tok.text for tok in doc]

relations = model.predict_entities(tokens, labels, threshold=0.5, ner=ner)

for rel in relations:
    print(f"{rel['head_text']} --> {rel['label']} --> {rel['tail_text']}")

import ipdb;ipdb.set_trace()

## Train

train_loader = model.create_dataloader(data, batch_size=5, shuffle=False)
iter_train_loader = iter(train_loader)
x = next(iter_train_loader)

loss = model(x)
loss