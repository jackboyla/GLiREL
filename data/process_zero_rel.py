from datasets import load_dataset, concatenate_datasets
import json
from tqdm import tqdm

NUM_TRAIN_EXAMPLES = 'all'
NUM_EVAL_EXAMPLES = 'all'

ds = None
for dataset_name in ["jackboyla/ZeroRel", 'jackboyla/gone_and_growned_my_own_dataset', 'jackboyla/zsre_grow']:
    dataset = load_dataset(dataset_name)    
    # features: ['id', 'text', 'tokenized_text', 'model_name', 'instruction', 'ents', 'generation', 'ner']
    if ds is None:
        ds = dataset['train']
    else:
        ds = concatenate_datasets([ds, dataset['train']])
    
print("Loaded datasets! Transforming...")
ds = ds.shuffle(seed=42)

# print("Asserting there's no duplicates... Removing if found...")
# seen_texts = set()
# selection_idx = []
# for i in tqdm(range(len(ds))):
#     if ds[i]['text'][0] not in seen_texts:
#         seen_texts.add(ds[i]['text'][0])
#         selection_idx.append(i)
# print(f"Found {len(ds) - len(selection_idx)} duplicates! :(")

# data = ds.select(selection_idx)
data = ds
data = data.to_dict()


def parse_generated_label(label: str):

    label = label.lower()
    label = label.replace('relation:', ' ')
    label = label.replace('\n', ' ')
    label = label.replace('\t', ' ')
    label = label.replace('.', ' ')
    label = label.replace(':', ' ')
    label = label.replace('"', ' ')
    label = label.replace("'", ' ')
    label = label.strip()
    return label.split(' ')[0]


def transform_zero_rel(data):
    # Transform the data into the desired format.
    transformed_data = []

    for i in tqdm(range(len(data['text']))):
        ner_entries = []
        relations = []
        tokens = data['tokenized_text'][i][0]

        assert len(data['generation'][i]) == len(data['ents'][i])

        for pair, relation_text in zip(data['ents'][i], data['generation'][i]):

            # Add head 
            head = pair[0]['head']
            head_start, head_end, head_type, head_text = int(head[0]), int(head[1]), head[2], head[3]
            ner_entries.append([head_start, head_end, head_type, head_text])
            
            # Add tail entity
            tail = pair[0]['tail']
            tail_start, tail_end, tail_type, tail_text = int(tail[0]), int(tail[1]), tail[2], tail[3]
            ner_entries.append([tail_start, tail_end, tail_type, tail_text])
            
            # Add relation
            relations.append({
                "head": {"mention": head_text, "position": [head_start, head_end], "type": head_type},
                "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": tail_type},
                # "relation_id": relation,
                "relation_text": parse_generated_label(relation_text),
            })

        transformed_data.append({
            "ner": ner_entries,
            "relations": relations,
            "tokenized_text": tokens,
        })

    return transformed_data


transformed_data = transform_zero_rel(data)

with open('./zero_rel_all.jsonl', 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item) + '\n')

