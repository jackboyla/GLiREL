from datasets import load_dataset
import json, statistics
from tqdm import tqdm
from random import randint

dataset = load_dataset("docred")    # features: ['relation', 'tokens', 'head', 'tail', 'names'],

SPLIT = 'validation'


'''

DocRED paper: https://arxiv.org/pdf/1906.06127.pdf
dataset: https://huggingface.co/datasets/thunlp/docred

all entities in DocRED are grouped together by co-reference resolution, 
and each group is treated as a single entity.

We want to split that up so that each mention is a separate entity.
This involves duplicating the relationships shared across the mentions of the same entity.
As well as creating a `SELF` relationship for each mention to mentions of the same entity.

Also getting the rest of the data in the right format for GLiREL training (ner, tokenized_text)

'''

def map_entities(doc):
    # Create a mapping for the entities
    entity_mapping = {}
    for entity_id, mentions in enumerate(doc['vertexSet']):
        for mention in mentions:
            position = (mention['pos'][0], mention['pos'][1])
            entity_mapping[(entity_id, mention['sent_id'], position)] = mention

    '''
    # key = (entity_id, sent_id, position)
    {(0, 4, (0, 2)): {'name': 'Skai TV', 'sent_id': 4, 'pos': [0, 2], 'type': 'ORG'},
    (0, 0, (0, 2)): {'name': 'Skai TV', 'sent_id': 0, 'pos': [0, 2], 'type': 'ORG'},
    ...
    '''
    return entity_mapping


def expand_labels(doc):
    expanded_labels = []
    labels = doc['labels']
    
    # Function to get cumulative position of an entity mention
    def get_cumulative_position(mention, cumulative_sentence_lengths):
        return cumulative_sentence_lengths[mention['sent_id']] + mention['pos'][0], \
               cumulative_sentence_lengths[mention['sent_id']] + mention['pos'][1]
    
    # Calculate cumulative sentence lengths for indexing
    cumulative_sentence_lengths = [0]
    for sentence in doc['sents']:
        cumulative_sentence_lengths.append(cumulative_sentence_lengths[-1] + len(sentence))
    
    # Iterate over the relations in the labels
    for i, (head_id, tail_id, relation_id, relation_text, evidence) in enumerate(zip(
        labels['head'], labels['tail'], labels['relation_id'], labels['relation_text'], labels['evidence'])):
        
        # Iterate over all mentions for the head entity
        for head_mention in doc['vertexSet'][head_id]:
            head_start_index, head_end_index = get_cumulative_position(head_mention, cumulative_sentence_lengths)
            
            # Iterate over all mentions for the tail entity
            for tail_mention in doc['vertexSet'][tail_id]:
                tail_start_index, tail_end_index = get_cumulative_position(tail_mention, cumulative_sentence_lengths)
                
                # Construct the relation for each mention
                expanded_relation = {
                    'head': {
                        'name': head_mention['name'],
                        'position': [head_start_index, head_end_index],
                        'type': head_mention['type']
                    },
                    'tail': {
                        'name': tail_mention['name'],
                        'position': [tail_start_index, tail_end_index],
                        'type': tail_mention['type']
                    },
                    'relation_id': relation_id,
                    'relation_text': relation_text,
                    'evidence': evidence
                }
                expanded_labels.append(expanded_relation)


    # Add SELF relations for mentions within the same entity cluster
    for entity_mentions in doc['vertexSet']:

        for i, head_mention in enumerate(entity_mentions):
            head_start_index, head_end_index = get_cumulative_position(head_mention, cumulative_sentence_lengths)

            for j, tail_mention in enumerate(entity_mentions):
                if i != j:  # Skip self-relation for the same mention
                    tail_start_index, tail_end_index = get_cumulative_position(tail_mention, cumulative_sentence_lengths)
                    
                    # Construct the SELF relation
                    self_relation = {
                        'head': {
                            'name': head_mention['name'],
                            'position': [head_start_index, head_end_index],
                            'type': head_mention['type']
                        },
                        'tail': {
                            'name': tail_mention['name'],
                            'position': [tail_start_index, tail_end_index],
                            'type': tail_mention['type']
                        },
                        'relation_id': 'SELF',
                        'relation_text': 'SELF',
                        'evidence': None  # no specific evidence for SELF relations
                    }
                    expanded_labels.append(self_relation)
    

    # sort for viewing
    expanded_labels = sorted(expanded_labels, key=lambda x: (x['head']['name'], x['head']['position'][0], x['tail']['position'][0]))
    return expanded_labels


def get_ner_from_entity_mapping(entity_mapping, doc):
    ner = []
    cumulative_offset = 0  
    for _, mention in entity_mapping.items():
        # Find the cumulative offset for the mention (add up the len of all sentences before the mention's sentence)
        cumulative_offset = sum(len(sent) for sent in doc['sents'][:mention['sent_id']])
        # Adjust the entity's position with the cumulative offset
        ner.append([cumulative_offset + mention['pos'][0], cumulative_offset + mention['pos'][1], mention['type'], mention['name']])
    ner = sorted(ner, key=lambda x: x[0])
    return ner


def assert_ner_aligns_with_text(ner, tokenized_text, i):
    for start, end, _, ner_surface_form in ner:

        target_start, target_end = ner_surface_form.split(' ')[0], ner_surface_form.split(' ')[-1]
        text_surface_form = tokenized_text[start:end]
        start, end = text_surface_form[0], text_surface_form[-1]
        assert target_start == start and target_end == end, f"Error in document {i} --> surface_form: {ner_surface_form} != Text: {text_surface_form}"


data = []
doc_lens = []
for i in tqdm(range(len(dataset[SPLIT]))):
    doc = dataset[SPLIT][i]
    doc_lens.append(sum(len(s) for s in doc['sents']))
    example_row = {}

    # Combine the list of sentences into one list of tokens
    example_row['tokenized_text'] = [token for sentence in doc['sents'] for token in sentence]

    entity_mapping = map_entities(doc)

    example_row['ner'] = get_ner_from_entity_mapping(entity_mapping, doc)  #  "ner": [[3,6,"LOC"], [7,9,"PER"]]
    # assert_ner_aligns_with_text(example_row['ner'], example_row['tokenized_text'], i)
    example_row['relations'] = expand_labels(doc)

    data.append(example_row)

# for ent in data[0]['ner']: print(f"{ent} => {data[0]['tokenized_text'][ent[0]:ent[1]]}")  # ner_sf => tokenized_text_sf
    
random_i = randint(0, len(data))
for rel in data[random_i]['relations']: print(f"{rel['head']['name']} {rel['head']['position']} -> {rel['relation_text']} -> {rel['tail']['name']} {rel['tail']['position']}")
    
print(f"Number of documents: {len(data)}")
print(f"Median length of documents: {statistics.median(doc_lens)}")
print(f"Max length of documents: {max(doc_lens)}")

# save to a jsonl file
with open('docred_expanded.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")