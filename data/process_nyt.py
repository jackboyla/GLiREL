# https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view?pli=1  <-- original dataset
# https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3  <-- CopyRE-processed dataset
# https://github.com/xiangrongzeng/copy_re?tab=readme-ov-file


import json
import random
import os
random.seed(12)
import zipfile

from tqdm import tqdm
import gdown

file_id = '1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N' 
output = 'raw_nyt.zip'
unzip_to = 'nyt'
if not os.path.exists(unzip_to):
    print("Downloading NYT dataset...")
    gdown.download(id=file_id, output=output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        # Extract all contents
        zip_ref.extractall(unzip_to)


# for all files in the NYT dataset, concat
data = []
for file in ['raw_train.json', 'raw_valid.json', 'raw_test.json']:
    with open(os.path.join(unzip_to, file), 'r') as f:
        data.extend([json.loads(l) for l in f])


'''
[
    {'sentText': 'But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer , the erratic former world chess champion , dished out in March at a news conference in Reykjavik , Iceland .', 
    'articleId': '/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1677367.xml.pb', 
    'relationMentions': [
            {'em1Text': 'Bobby Fischer', 'em2Text': 'Iceland', 'label': '/people/person/nationality'}, 
            {'em1Text': 'Iceland', 'em2Text': 'Reykjavik', 'label': '/location/country/capital'}, 
            {'em1Text': 'Iceland', 'em2Text': 'Reykjavik', 'label': '/location/location/contains'}, 
            {'em1Text': 'Bobby Fischer', 'em2Text': 'Reykjavik', 'label': '/people/deceased_person/place_of_death'}], 
    'entityMentions': [
            {'start': 0, 'label': 'PERSON', 'text': 'Bobby Fischer'}, 
            {'start': 1, 'label': 'LOCATION', 'text': 'Reykjavik'}, 
            {'start': 2, 'label': 'LOCATION', 'text': 'Iceland'}], 'sentId': '1'}
]
'''


import re
import json
from typing import List, Dict, Any
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text by removing diacritics (accents) and converting to lowercase.
    """
    text_nfkd = unicodedata.normalize('NFKD', text)
    text_ascii = ''.join([c for c in text_nfkd if not unicodedata.combining(c)])
    return text_ascii

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer that splits text into tokens based on whitespace and punctuation.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def find_entity_span(tokens: List[str], entity_text: str) -> List[int]:
    """
    Finds the start and end token indices for a given entity_text within the tokens list.
    Returns a list [start, end]. If not found, returns [-1, -1].
    """
    entity_tokens = tokenize(entity_text)
    n = len(entity_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == entity_tokens:
            # Assertion to ensure the span matches the entity text
            matched_text = ''.join(tokens[i:i+n]).replace(" ", "")
            entity_text_clean = entity_text.replace(" ", "")
            assert matched_text == entity_text_clean, (
                f"Token span '{tokens[i:i+n]}' does not match entity text '{entity_text}'."
            )
            return [i, i + n - 1]
    return [-1, -1]

def map_relations(labels: List[str]) -> Dict[str, str]:
    """
    Maps unique relation labels to unique relation IDs.
    For simplicity, assigns 'P' followed by a unique number starting from 1.
    """
    unique_labels = sorted(set(labels))
    label_to_id = {label: f"{idx+1}" for idx, label in enumerate(unique_labels)}
    return label_to_id

def convert_entry(entry: Dict[str, Any], relation_id_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Converts a single entry to the desired format with added assertions for entity alignment.
    """
    sent_text = entry['sentText']
    tokens = tokenize(sent_text)
    
    # Process NER
    ner = []
    entity_spans = {}
    for idx, entity in enumerate(entry.get('entityMentions', [])):
        entity_text = entity['text']
        label = entity['label']
        span = find_entity_span(tokens, entity_text)
        if span[0] == -1:
            print(f"Warning: Entity '{entity_text}' not found in tokens.")
            import ipdb; ipdb.set_trace()
            continue
        # Additional check to ensure the tokens match the entity text
        extracted_text = ' '.join(tokens[span[0]:span[1]+1])
        if extracted_text.replace(" ", "") != entity_text.replace(" ", ""):
            raise ValueError(
                f"Entity text mismatch for '{entity_text}'. "
                f"Extracted: '{extracted_text}', Expected: '{entity_text}'."
            )
        ner.append([span[0], span[1], label, entity_text])
        entity_spans[entity_text] = {
            "mention": entity_text,
            "position": span,
            "type": label
        }
    
    # Process Relations
    relations = []
    for relation in entry.get('relationMentions', []):
        em1_text = relation['em1Text']
        em2_text = relation['em2Text']
        label = relation['label']

        em1_text = normalize_text(em1_text)
        em2_text = normalize_text(em2_text)
        
        head = entity_spans.get(em1_text)
        tail = entity_spans.get(em2_text)
        
        if not head or not tail:
            print(f"Warning: Relation entities '{em1_text}' or '{em2_text}' not found.")
            import ipdb; ipdb.set_trace()
            continue
        
        # Extract relation_text from label
        relation_text = label.split('/')[-1] if '/' in label else label
        relation_id = relation_id_map.get(label, "P0")  # Default to "P0" if not found
        
        relations.append({
            "head": head,
            "tail": tail,
            "relation_id": relation_id,
            "relation_text": relation_text
        })
    
    return {
        "ner": ner,
        "relations": relations,
        "tokenized_text": tokens
    }

def convert_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts a list of entries to the desired format with entity alignment checks.
    """
    # Collect all relation labels to create a mapping
    all_labels = []
    for entry in data:
        for rel in entry.get('relationMentions', []):
            all_labels.append(rel['label'])
    
    relation_id_map = map_relations(all_labels)
    
    converted = []
    for entry in tqdm(data, desc="Converting data"):
        try:
            converted_entry = convert_entry(entry, relation_id_map)
            converted.append(converted_entry)
        except AssertionError as ae:
            print(f"AssertionError: {ae}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
    
    return converted


nyt_converted = convert_data(data)


save_path = './nyt/nyt_all.jsonl'
with open(save_path, 'w') as f:
    for item in nyt_converted:
        f.write(json.dumps(item) + '\n')
print(f"Saved to {save_path}")

