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
    label = label.strip()

    return label


def transform_zero_rel(data):
    # Transform the data into the desired format.
    transformed_data = []

    for i in tqdm(range(len(data['text']))):

        ## NOTE: spacy will index entities [start(inclusive), end(exclusive)]
        # e.g ["The", "quick", "brown", "fox"] --> "quick" is [1, 2]
        # Our model expects [start(inclusive), end(inclusive)], hence the -1's below

        ner_entries = ([[int(ent[0]), int(ent[1]) - 1, ent[2], ent[3]] for ent in data['ner'][i]])
        relations = []
        tokens = data['tokenized_text'][i][0]
        seen_rels = set()

        assert len(data['generation'][i]) == len(data['ents'][i])

        for pair, relation_text in zip(data['ents'][i], data['generation'][i]):

            # Add head 
            head = pair[0]['head']
            head_start, head_end, head_type, head_text = int(head[0]), int(head[1]) - 1, head[2], head[3]
            
            # Add tail entity
            tail = pair[0]['tail']
            tail_start, tail_end, tail_type, tail_text = int(tail[0]), int(tail[1]) - 1, tail[2], tail[3]
            
            # Add relation
            relations.append({
                "head": {"mention": head_text, "position": [head_start, head_end], "type": head_type},
                "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": tail_type},
                "relation_text": parse_generated_label(relation_text),
                "raw_relation_text": relation_text,
            })
            seen_rels.add(((head_start, head_end), (tail_start, tail_end)))

        # fill empty relations with "no relation"
        for head in ner_entries:
            head_start, head_end, head_type, head_text = int(head[0]), int(head[1]), head[2], head[3]
            for tail in ner_entries:
                tail_start, tail_end, tail_type, tail_text = int(tail[0]), int(tail[1]), tail[2], tail[3]

                if (head_start, head_end) != (tail_start, tail_end) and ((head_start, head_end), (tail_start, tail_end)) not in seen_rels:
                    relations.append({
                        "head": {"mention": head_text, "position": [head_start, head_end], "type": head_type},
                        "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": tail_type},
                        "relation_text": "no relation",
                        "raw_relation_text": "no relation",
                    })

        transformed_data.append({
            "ner": ner_entries,
            "relations": relations,
            "tokenized_text": tokens,
        })

    return transformed_data


save_path = './zero_rel_all.jsonl'
with open(save_path, 'w') as f:
    step_size = 10_000
    for step in range(0, len(ds), step_size):
        end = min(step + step_size, len(ds))
        data = ds.select(range(step, end))
        data = data.to_dict()
        transformed_data = transform_zero_rel(data)

        for item in transformed_data:
            f.write(json.dumps(item) + '\n')
print(f"Saved to {save_path}")


# post processing #########################
print("Post processing...")
print(f"Assigning 'no relation' label to relations with troublesome labels... (see process_zero_rel.py)")

with open(save_path, 'r') as f:
    data = [json.loads(line) for line in f]

relationship_counts = {}
raw_relationship_string = {}

for item in data:
    relations = item['relations']
    for relation in relations:
        relation_text = relation['relation_text']
        if relation_text in relationship_counts:
            relationship_counts[relation_text] += 1
        else:
            relationship_counts[relation_text] = 1

        raw_relation = relation['raw_relation_text']
        if relation_text not in raw_relationship_string:
            raw_relationship_string[relation_text] = set()
        raw_relationship_string[relation_text].add(relation['raw_relation_text'])


reassign_count = 0
for item in tqdm(data):
    relations = item['relations']
    for relation in relations:
        rel_text = relation['relation_text']
        if len(rel_text) < 4 and relationship_counts[rel_text] < 200:
            relation['relation_text'] = 'no relation'
            reassign_count += 1
        elif len(rel_text) < 2:
            relation['relation_text'] = 'no relation'
            reassign_count += 1
        elif relationship_counts[rel_text] < 10:
            relation['relation_text'] = 'no relation'
            reassign_count += 1

print(f"Reassigned {reassign_count} relations to 'no relation'")
########################################
    

with open(save_path, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
print(f"Saved to {save_path}")
