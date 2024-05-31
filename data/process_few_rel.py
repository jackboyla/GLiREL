from datasets import load_dataset, concatenate_datasets
import json

NUM_TRAIN_EXAMPLES = 'all'
NUM_EVAL_EXAMPLES = 'all'

dataset = load_dataset("few_rel")    # features: ['relation', 'tokens', 'head', 'tail', 'names'],
ds_train = dataset['train_wiki'].shuffle(seed=42)
ds_val = dataset['val_wiki'].shuffle(seed=42)
ds = concatenate_datasets([ds_train, ds_val])
print(f"Number of examples: {len(ds)}")

# for i in range(15): 
#     rel_texts = [rel for rel in ds[i]['names'][i]]
#     print(f"Relation: {rel_texts}")

if type(NUM_TRAIN_EXAMPLES) is int:
    data = ds.select(range(NUM_TRAIN_EXAMPLES))
else:
    data = ds
data = data.to_dict()

'''
one relation, two entities
names has the text of the entities for head and tail respectively
need to get relation text from Wikidata
{
    'relation': ['P931', 'P931', 'P931'], 
    'tokens': [['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], ['The', 'name', 'was', 'at', 'one', 'point', 'changed', 'to', 'Nottingham', 'East', 'Midlands', 'Airport', 'so', 'as', 'to', 'include', 'the', 'name', 'of', 'the', 'city', 'that', 'is', 'supposedly', 'most', 'internationally', 'recognisable', ',', 'mainly', 'due', 'to', 'the', 'Robin', 'Hood', 'legend', '.'], [
        'It', 'is', 'a', 'four', '-', 'level', 'stack', 'interchange', 'near', 'Fort', 'Lauderdale', '-', 'Hollywood', 'International', 'Airport', 'in', 'Fort', 'Lauderdale', ',', 'Florida', '.']], 
        'head': [{'text': 'tjq', 'type': 'Q1331049', 'indices': [[16]]}, {'text': 'east midlands airport', 'type': 'Q8977', 'indices': [[9, 10, 11]]}, {'text': 'fort lauderdale-hollywood international airport', 'type': 'Q635361', 'indices': [[9, 10, 11, 12, 13, 14]]}], 
        'tail': [{'text': 'tanjung pandan', 'type': 'Q3056359', 'indices': [[13, 14]]}, {'text': 'nottingham', 'type': 'Q41262', 'indices': [[8]]}, {'text': 'fort lauderdale, florida', 'type': 'Q165972', 'indices': [[16, 17, 18, 19]]}], 
        'names': [['place served by transport hub', 'territorial entity or entities served by this transport hub (airport, train station, etc.)'], ['place served by transport hub', 'territorial entity or entities served by this transport hub (airport, train station, etc.)'], ['place served by transport hub', 'territorial entity or entities served by this transport hub (airport, train station, etc.)']]
}
'''

def transform_few_rel(data):
    # Transform the data into the desired format.
    transformed_data = []

    for i in range(len(data['relation'])):
        ner_entries = []
        relations = []
        tokens = data['tokens'][i]
        head = data['head'][i]
        tail = data['tail'][i]
        relation = data['relation'][i]
        relation_text = data['names'][i][0]

        # Add head entity
        head_start, head_end = head['indices'][0][0], head['indices'][0][-1] + 1
        head_text = " ".join(tokens[head_start:head_end])
        ner_entries.append([head_start, head_end, head['type'], head_text])
        
        # Add tail entity
        tail_start, tail_end = tail['indices'][0][0], tail['indices'][0][-1] + 1
        tail_text = " ".join(tokens[tail_start:tail_end])
        ner_entries.append([tail_start, tail_end, tail['type'], tail_text])
        
        # Add relation
        relations.append({
            "head": {"mention": head_text, "position": [head_start, head_end], "type": head['type']},
            "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": tail['type']},
            "relation_id": relation,
            "relation_text": relation_text,
        })

        transformed_data.append({
            "ner": ner_entries,
            "relations": relations,
            "tokenized_text": tokens,
        })

    return transformed_data

transformed_data = transform_few_rel(data)

with open('./few_rel_all.jsonl', 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item) + '\n')
