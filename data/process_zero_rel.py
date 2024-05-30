from datasets import load_dataset, concatenate_datasets
import json
from tqdm import tqdm

SEED = 42

NUM_TRAIN_EXAMPLES = 'all'
NUM_EVAL_EXAMPLES = 'all'

ds = None
for dataset_name in ["jackboyla/ZeroRel", 'jackboyla/gone_and_growned_my_own_dataset', 'jackboyla/zsre_grow']: #  
    dataset = load_dataset(dataset_name, download_mode='force_redownload')    
    # features: ['id', 'text', 'tokenized_text', 'model_name', 'instruction', 'ents', 'generation', 'ner']
    if ds is None:
        ds = dataset['train']
    else:
        ds = concatenate_datasets([ds, dataset['train']])
    
print("Loaded datasets! Transforming...")
ds = ds.shuffle(seed=SEED)

# print("Asserting there's no duplicates... Removing if found...")
# seen_texts = set()
# selection_idx = []
# for i in tqdm(range(len(ds))):
#     if ds[i]['text'][0] not in seen_texts:
#         seen_texts.add(ds[i]['text'][0])
#         selection_idx.append(i)
# print(f"Found {len(ds) - len(selection_idx)} duplicates! :(")

# data = ds.select(selection_idx)


def parse_generated_label(label: str):

    label = label.lower()
    label = label.replace('relation:', ' ')
    label = label.replace('\n', ' ')
    label = label.replace('\t', ' ')
    label = label.replace('.', ' ')
    label = label.replace(':', ' ')
    label = label.replace('"', ' ')
    label = label.replace("'", ' ')
    label = label.replace("{", ' ')
    label = label.replace("}", ' ')
    label = label.replace("[", ' ')
    label = label.replace("]", ' ')
    label = label.strip()
    label = label.split(' ')[0]
    label = label.replace('_', " ")
    return label


def transform_zero_rel(data):
    # Transform the data into the desired format.
    transformed_data = []

    for i in tqdm(range(len(data['text']))):
        ner_entries = ([[int(ent[0]), int(ent[1]), ent[2], ent[3]] for ent in data['ner'][i]])
        relations = []
        tokens = data['tokenized_text'][i][0]
        seen_rels = set()

        assert len(data['generation'][i]) == len(data['ents'][i])

        for pair, relation_text in zip(data['ents'][i], data['generation'][i]):

            # Add head 
            head = pair[0]['head']
            head_start, head_end, head_type, head_text = int(head[0]), int(head[1]), head[2], head[3]
            
            # Add tail entity
            tail = pair[0]['tail']
            tail_start, tail_end, tail_type, tail_text = int(tail[0]), int(tail[1]), tail[2], tail[3]
            
            # Add relation
            relations.append({
                "head": {"mention": head_text, "position": [head_start, head_end], "type": head_type},
                "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": tail_type},
                "relation_text": parse_generated_label(relation_text),
            })
            seen_rels.add(((head_start, head_end), (tail_start, tail_end)))

        # fill empty relations with "no relation"
        for head_start, head_end, head_type, head_text in ner_entries:
            # head_start, head_end, head_type, head_text = int(ent1[0]), int(ent1[1]), ent1[2], ent1[3]
            for tail_start, tail_end, tail_type, tail_text in ner_entries:
                # tail_start, tail_end, tail_type, tail_text = int(ent2[0]), int(ent2[1]), ent2[2], ent2[3]

                if (head_start, head_end) != (tail_start, tail_end) and ((head_start, head_end), (tail_start, tail_end)) not in seen_rels:
                    relations.append({
                        "head": {"mention": head_text, "position": [head_start, head_end], "type": head_type},
                        "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": tail_type},
                        "relation_text": parse_generated_label(relation_text),
                    })

        transformed_data.append({
            "ner": ner_entries,
            "relations": relations,
            "tokenized_text": tokens,
        })

    return transformed_data


with open('./zero_rel_all.jsonl', 'w') as f:
    step_size = 10_000
    for step in range(0, len(ds), step_size):
        end = min(step + step_size, len(ds))
        data = ds.select(range(step, end))
        data = data.to_dict()
        transformed_data = transform_zero_rel(data)

        for item in transformed_data:
            f.write(json.dumps(item) + '\n')

