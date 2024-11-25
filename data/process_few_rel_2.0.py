from datasets import load_dataset, concatenate_datasets
import json
import requests


def download_dataset(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    data = json.loads(response.content)
    return data

# URLs for the datasets
train_url = 'https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json'
eval_url = 'https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json'

train_data = download_dataset(train_url)
eval_data = download_dataset(eval_url)


# combine the train and eval data
'''
train_data.keys()
dict_keys(['P931', 'P4552', 'P140', 'P1923', 'P150', 'P6', 'P27', 'P449', 'P1435', 'P175', 'P1344', 'P39', 'P527', 'P740', 'P706', 'P84', 'P495', 'P123', 'P57', 'P22', 'P178', 'P241', 'P403', 'P1411', 'P135', 'P991', 'P156', 'P176', 'P31', 'P1877', 'P102', 'P1408', 'P159', 'P3373', 'P1303', 'P17', 'P106', 'P551', 'P937', 'P355', 'P710', 'P137', 'P674', 'P466', 'P136', 'P306', 'P127', 'P400', 'P974', 'P1346', 'P460', 'P86', 'P118', 'P264', 'P750', 'P58', 'P3450', 'P105', 'P276', 'P101', 'P407', 'P1001', 'P800', 'P131'])
'''
data = {}
for dataset in [train_data, eval_data]:
    for key in dataset.keys():
        if key in data:
            data[key] += dataset[key]
        else:
            data[key] = dataset[key]

# load PID2Name mapping
with open('all_wikidata_properties.json', 'r') as f:
    properties = json.load(f)      
    pid2name = {}
    for property in properties:
        pid = property['property'].split('/')[-1]
        pid2name[pid] = [property['propertyLabel']]

pid2name.update(json.load(open('pid2name_fewrel.json')))
pid2name.update(json.load(open('pid2name_wiki.json')))


def _generate_examples(data, pid2name: dict):
    """Format into `hf_datset.dict()` format. This is convenient for re-using the process_fewl_rel.py script"""
    id = 0
    relation_list = []
    tokens_list = []
    head_list = []
    tail_list = []
    names_list = []
    for key in list(data.keys()):
        # e.g. key = 'P931'
        for items in data[key]:
            tokens = items["tokens"]
            h_0 = items["h"][0]
            h_1 = items["h"][1]
            h_2 = items["h"][2]
            t_0 = items["t"][0]
            t_1 = items["t"][1]
            t_2 = items["t"][2]
            id += 1
            # yield id, {
            #     "relation": key,
            #     "tokens": tokens,
            #     "head": {"text": h_0, "type": h_1, "indices": h_2},
            #     "tail": {"text": t_0, "type": t_1, "indices": t_2},
            #     "names": [key],
            # }
            relation_list.append(key)
            tokens_list.append(tokens)
            head_list.append({"text": h_0, "type": h_1, "indices": h_2})
            tail_list.append({"text": t_0, "type": t_1, "indices": t_2})
            names_list.append(pid2name[key])

    return {
        "relation": relation_list,
        "tokens": tokens_list,
        "head": head_list,
        "tail": tail_list,
        "names": names_list,
    }


data = _generate_examples(data, pid2name)

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
        head_start, head_end = head['indices'][0][0], head['indices'][0][-1]
        head_text = " ".join(tokens[head_start:head_end+1])
        ner_entries.append([head_start, head_end, head['type'], head_text])
        
        # Add tail entity
        tail_start, tail_end = tail['indices'][0][0], tail['indices'][0][-1]
        tail_text = " ".join(tokens[tail_start: tail_end+1])
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

save_path = './few_rel_2.0_all.jsonl'
with open(save_path, 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item) + '\n')
print(f"Saved data to {save_path}")
