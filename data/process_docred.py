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
                        'type': head_mention['type'],
                        'h_idx': head_id
                    },
                    'tail': {
                        'name': tail_mention['name'],
                        'position': [tail_start_index, tail_end_index],
                        'type': tail_mention['type'],
                        't_idx': tail_id
                    },
                    'relation_id': relation_id,
                    'relation_text': relation_text,
                    'evidence': evidence
                }
                expanded_labels.append(expanded_relation)


    # Add SELF relations for mentions within the same entity cluster
    for cluster_idx, entity_mentions in enumerate(doc['vertexSet']):

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
                            'type': head_mention['type'],
                            'h_idx': cluster_idx
                        },
                        'tail': {
                            'name': tail_mention['name'],
                            'position': [tail_start_index, tail_end_index],
                            'type': tail_mention['type'],
                            't_idx': cluster_idx
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


for SPLIT in ['train_annotated', 'validation', 'test']: # train_distant
    data = []
    doc_lens = []
    for i in tqdm(range(len(dataset[SPLIT]))):
        doc = dataset[SPLIT][i]
        doc_lens.append(sum(len(s) for s in doc['sents']))
        example_row = {'title': doc['title']}

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
    # relation statistics
    relation_counts = []
    span_counts = []
    for doc in data:
        relation_counts.append([rel['relation_text'] for rel in doc['relations']])
        span_counts.append(doc['ner'])
    print(f"Median number of relations per document: {statistics.median([len(rels) for rels in relation_counts])}")
    print(f"Max number of relations per document: {max([len(rels) for rels in relation_counts])}")
    print(f"Median number of spans per document: {statistics.median([len(spans) for spans in span_counts])}")
    print(f"Max number of spans per document: {max([len(spans) for spans in span_counts])}")

    # save to a jsonl file
    with open(f'docred_{SPLIT}.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")





'''
X-Files [152, 155] -> country of origin -> American [14, 15]
X-Files [152, 155] -> SELF -> X - Files [20, 23]
X-Files [152, 155] -> original network -> Fox [28, 29]
X-Files [152, 155] -> start time -> December   15 ,   1996 [31, 37]
X-Files [152, 155] -> director -> Rob Bowman [47, 49]
X-Files [152, 155] -> cast member -> Tom Noonan [55, 57]
X-Files [152, 155] -> cast member -> Vanessa Morley [61, 63]
X-Files [152, 155] -> cast member -> Noonan [116, 117]
X-Files [152, 155] -> characters -> Fox Mulder [128, 130]
X-Files [152, 155] -> cast member -> David Duchovny [131, 133]
X-Files [152, 155] -> characters -> Dana Scully [135, 137]
X-Files [152, 155] -> cast member -> Gillian Anderson [138, 140]
X-Files [152, 155] -> characters -> Mulder [156, 157]
X-Files [152, 155] -> characters -> Scully [167, 168]
X-Files [152, 155] -> characters -> Mulder [180, 181]
X-Files [152, 155] -> characters -> Scully [182, 183]
X-Files [152, 155] -> cast member -> Tom Noonan [189, 191]
X-Files [152, 155] -> characters -> Mulder [193, 194]
X-Files [152, 155] -> characters -> Mulder [227, 228]
X-Files [152, 155] -> cast member -> Tom Noonan [293, 295]
Number of documents: 3053
Median length of documents: 179
Max length of documents: 511
Median number of relations per document: 47
Max number of relations per document: 1162
Median number of spans per document: 25
Max number of spans per document: 63
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 998/998 [00:01<00:00, 551.89it/s]
Baroque Churches of the Philippines [100, 105] -> instance of -> UNESCO World Heritage Sites [94, 98]
Boljo-on [29, 32] -> SELF -> Boljoon [45, 46]
Boljoon [45, 46] -> SELF -> Boljo-on [29, 32]
Boljoon Church [85, 87] -> located in the administrative territorial entity -> Boljo-on [29, 32]
Boljoon Church [85, 87] -> located in the administrative territorial entity -> Boljoon [45, 46]
Boljoon Church [85, 87] -> SELF -> Boljoon Church [131, 133]
Boljoon Church [131, 133] -> located in the administrative territorial entity -> Boljo-on [29, 32]
Boljoon Church [131, 133] -> located in the administrative territorial entity -> Boljoon [45, 46]
Boljoon Church [131, 133] -> SELF -> Boljoon Church [85, 87]
UNESCO [120, 121] -> SELF -> UNESCO [144, 145]
UNESCO [144, 145] -> SELF -> UNESCO [120, 121]
Number of documents: 998
Median length of documents: 181.0
Max length of documents: 510
Median number of relations per document: 47.0
Max number of relations per document: 1432
Median number of spans per document: 26.0
Max number of spans per document: 62
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 914.22it/s]
America [29, 30] -> SELF -> the United States [14, 17]
America [29, 30] -> SELF -> America [237, 238]
America [237, 238] -> SELF -> the United States [14, 17]
America [237, 238] -> SELF -> America [29, 30]
Blue Ridge Parkway [1, 4] -> SELF -> Blue Ridge Parkway [149, 152]
Blue Ridge Parkway [149, 152] -> SELF -> Blue Ridge Parkway [1, 4]
Great Smoky Mountains National Park [50, 55] -> SELF -> Great Smoky Mountains National Park [90, 95]
Great Smoky Mountains National Park [90, 95] -> SELF -> Great Smoky Mountains National Park [50, 55]
National Park Service [135, 138] -> SELF -> National Park Service [208, 211]
National Park Service [208, 211] -> SELF -> National Park Service [135, 138]
North Carolina [41, 43] -> SELF -> North Carolina [101, 103]
North Carolina [41, 43] -> SELF -> North Carolina [231, 233]
North Carolina [101, 103] -> SELF -> North Carolina [41, 43]
North Carolina [101, 103] -> SELF -> North Carolina [231, 233]
North Carolina [231, 233] -> SELF -> North Carolina [41, 43]
North Carolina [231, 233] -> SELF -> North Carolina [101, 103]
Shenandoah National Park [46, 49] -> SELF -> Shenandoah National Park [110, 113]
Shenandoah National Park [110, 113] -> SELF -> Shenandoah National Park [46, 49]
Skyline Drive [122, 124] -> SELF -> Skyline Drive [141, 143]
Skyline Drive [141, 143] -> SELF -> Skyline Drive [122, 124]
Virginia [39, 40] -> SELF -> Virginia [114, 115]
Virginia [39, 40] -> SELF -> Virginia [145, 146]
Virginia [114, 115] -> SELF -> Virginia [39, 40]
Virginia [114, 115] -> SELF -> Virginia [145, 146]
Virginia [145, 146] -> SELF -> Virginia [39, 40]
Virginia [145, 146] -> SELF -> Virginia [114, 115]
the United States [14, 17] -> SELF -> America [29, 30]
the United States [14, 17] -> SELF -> America [237, 238]
Number of documents: 1000
Median length of documents: 181.0
Max length of documents: 508
Median number of relations per document: 20.0
Max number of relations per document: 308
Median number of spans per document: 26.0
Max number of spans per document: 64
'''