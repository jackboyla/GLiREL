from datasets import load_dataset, concatenate_datasets
import datasets
import json
from tqdm import tqdm

SEED = 42

NUM_TRAIN_EXAMPLES = 'all'
NUM_EVAL_EXAMPLES = 'all'

print("Loading datasets...")
dataset_names = ["jackboyla/ZeroRel"]

# Initialize an empty list to store datasets
datasets_list = []

for dataset_name in dataset_names:
    dataset = load_dataset(dataset_name, split='train', streaming=True)
    datasets_list.append(dataset)

# Concatenate datasets using interleave (works with streaming datasets)
ds = datasets.interleave_datasets(datasets_list, seed=SEED)

# Shuffle the dataset (shuffling streaming datasets shuffles the order of items)
ds = ds.shuffle(buffer_size=10_000, seed=SEED)

print("Loaded and shuffled datasets! Transforming...")

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

def transform_zero_rel_example(example):
    # Transform a single example into the desired format.

    # Extract necessary fields
    text = example['text']
    tokenized_text = example['tokenized_text']
    ner_entries = [[int(ent[0]), int(ent[1]) - 1, ent[2], ent[3]] for ent in example['ner']]
    generation = example['generation']
    ents = example['ents']

    tokens = tokenized_text[0]
    seen_rels = set()

    relations = []

    assert len(generation) == len(ents)

    for pair, relation_text in zip(ents, generation):
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

    # Fill empty relations with "no relation"
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

    transformed_example = {
        "ner": ner_entries,
        "relations": relations,
        "tokenized_text": tokens,
    }

    return transformed_example

save_path = './zero_rel_intermediate.jsonl'

print("Processing and saving transformed data...")
with open(save_path, 'w') as f:
    for example in tqdm(ds):
        transformed_example = transform_zero_rel_example(example)
        f.write(json.dumps(transformed_example) + '\n')

print(f"Saved transformed data to {save_path}")

# Post-processing
print("Post-processing...")

# Initialize counters and mappings
relationship_counts = {}
raw_relationship_strings = {}

# First pass: calculate relationship counts
print("Calculating relationship counts...")
with open(save_path, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        relations = item['relations']
        for relation in relations:
            relation_text = relation['relation_text']
            relationship_counts[relation_text] = relationship_counts.get(relation_text, 0) + 1

            raw_relation = relation['raw_relation_text']
            if relation_text not in raw_relationship_strings:
                raw_relationship_strings[relation_text] = set()
            raw_relationship_strings[relation_text].add(raw_relation)

# Second pass: reassign labels and write to a new file
print("Reassigning labels and writing final data...")
reassign_count = 0
final_save_path = './zero_rel_all.jsonl'

with open(save_path, 'r') as fr, open(final_save_path, 'w') as fw:
    for line in tqdm(fr):
        item = json.loads(line)
        relations = item['relations']
        for relation in relations:
            rel_text = relation['relation_text']
            count = relationship_counts.get(rel_text, 0)
            # Apply reassigning conditions
            if (len(rel_text) < 4 and count < 200) or len(rel_text) < 2 or count < 10:
                if relation['relation_text'] != 'no relation':
                    relation['relation_text'] = 'no relation'
                    reassign_count += 1
                    relationship_counts['no relation'] += 1

        # Write the updated item to the new file
        fw.write(json.dumps(item) + '\n')

# save relation type counts
print(f"Relationship counts: {relationship_counts}")
with open(f"./zero_rel_type_counts.json", "w") as f:
    f.write(json.dumps(relationship_counts))

print(f"Reassigned {reassign_count} relations to 'no relation'")
print(f"Saved final processed data to {final_save_path}")

