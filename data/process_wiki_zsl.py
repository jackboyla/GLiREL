import json
import random
random.seed(12)

NUM_EXAMPLES = 'all'

import gdown

print("Downloading Wiki_ZSL dataset...")
url = 'https://drive.google.com/uc?id=1ELFGUIYDClmh9GrEHjFYoE_VI1t2a5nK'
output = 'wiki_all.json'
gdown.download(url, output, quiet=False)


with open('wiki_all.json', 'r') as f:    # len --> 93483
    dataset = json.load(f)

with open('all_wikidata_properties.json', 'r') as f:
    properties = json.load(f)      
    pid2name = {}
    for property in properties:
        pid = property['property'].split('/')[-1]
        pid2name[pid] = property['propertyLabel']

pid2name.update(json.load(open('pid2name_fewrel.json')))
pid2name.update(json.load(open('pid2name_wiki.json')))

# NEW / CHANGED PROPERTIES
# P7 (brother) --> P3373 (sibling)
pid2name['P7'] = ["sibling", "the subject and the object have at least one common parent (brother, sister, etc. including half-siblings)"]
# P9 (sister) --> P3373 (sibling)
pid2name['P9'] = ["sibling", "the subject and the object have at least one common parent (brother, sister, etc. including half-siblings)"]

'''
{
    "P17": [
        "country",
        "sovereign state of this item; don't use on humans"
    ],
}
'''


if type(NUM_EXAMPLES) is int:
    data = dataset[:NUM_EXAMPLES]
else:
    data = dataset

'''

{
    'vertexSet': [
        {'kbID': '1992-01-01', 'type': 'DATE', 'lexicalInput': '1992-01-01', 'unique': False, 'namedEntity': False, 'numericalValue': 0.0, 
        'dateValue': {'year': 1992, 'month': 1, 'day': 1}, 
        'tokenpositions': [17], 'variable': False, 'pos': 'CD'}, 
        {'kbID': 'Q759874', 'type': 'LEXICAL', 'variable': False, 'unique': False, 'namedEntity': True, 
        'tokenpositions': [10, 11], 'numericalValue': 0.0, 'lexicalInput': 'Habitats Directive'}, 
        {'kbID': 'Q458', 'type': 'LEXICAL', 'variable': False, 'unique': False, 'namedEntity': True, 
        'tokenpositions': [14, 15], 'numericalValue': 0.0, 'lexicalInput': 'European Union'}, 
        {'kbID': 'Q1191622', 'type': 'LEXICAL', 'variable': False, 'unique': False, 'namedEntity': True, 
        'tokenpositions': [26, 27, 28, 29], 'numericalValue': 0.0, 'lexicalInput': 'special areas of conservation'}
    ], 
    'edgeSet': [
        {'left': [26, 27, 28, 29], 'right': [14, 15], 'kbID': 'P1001'}, 
        {'kbID': 'P0', 'right': [14, 15], 'left': [10, 11]}], 
    'tokens': ['L.', 'cervus', 'is', 'registered', 'in', 'the', 'second', 'appendix', 'of', 'the', 'Habitats', 'Directive', 'of', 'the', 'European', 'Union', 'from', '1992', ',', 'which', 'requires', 'that', 'member', 'states', 'set', 'aside', 'special', 'areas', 'of', 'conservation', '.']
}

'''

def transform_wiki_zsl(dataset):

    transformed_data = []
    problem_entities = []
    problem_relations = []

    for data in dataset:

        tokens = data['tokens']

        # Prepare to collect NER entries
        ner_entries = []
        for vertex in data['vertexSet']:
            if vertex.get('tokenpositions'):
                start = vertex['tokenpositions'][0]
                end = vertex['tokenpositions'][-1] + 1  # Include last token
                entity_text = " ".join(tokens[start:end])
                ner_entries.append([start, end, vertex['type'], entity_text])
            else:
                problem_entities.append(vertex)
                # # Attempt to find the entity by lexicalInput
                # lexical_input = vertex.get('lexicalInput', '')
                # # Normalize and split the lexicalInput and the token list for comparison
                # lex_input_tokens = lexical_input.split()
                # # Search for the exact sequence of words in tokens
                # try:
                #     # Join tokens and use the .index() method to find the starting index of the joined lexical_input
                #     combined_tokens = " ".join(tokens)
                #     start_index = combined_tokens.index(lexical_input)
                #     # Convert start index in characters back to start index in tokens
                #     start = len(combined_tokens[:start_index].split())
                #     end = start + len(lex_input_tokens)
                #     entity_text = " ".join(tokens[start:end])
                #     ner_entries.append([start, end, vertex['type'], entity_text])
                # except ValueError:
                #     print(f"Lexical input not found in tokens: {lexical_input}")
                #     problem_entities.append(vertex)

        
        # Prepare to collect relations
        relations = []
        for edge in data['edgeSet']:
            if edge['kbID'] == 'P0':  # Skip P0 (None) relations
                continue
            if len(edge['left']) == 0 or len(edge['right']) == 0: # Skip relations with no positions
                problem_relations.append(edge)
                continue
            if edge['kbID'] not in pid2name:   # Skip relations not found in pid2name
                print('Relation not found:', edge['kbID'])
                continue
            head_start = edge['left'][0]
            head_end = edge['left'][-1] + 1
            tail_start = edge['right'][0]
            tail_end = edge['right'][-1] + 1
            head_text = " ".join(tokens[head_start:head_end])
            tail_text = " ".join(tokens[tail_start:tail_end])

            # Find and append relation
            relations.append({
                "head": {"mention": head_text, "position": [head_start, head_end], "type": find_type_by_position(data, head_start, head_end)},
                "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": find_type_by_position(data, tail_start, tail_end)},
                "relation_id": edge['kbID'],
                "relation_text": pid2name[edge['kbID']][0],
            })

        # ensure that all positions in relations exist in ner
        for relation in relations:
            head_start, head_end = relation['head']['position']
            tail_start, tail_end = relation['tail']['position']
            if not any([head_start == start and head_end == end for start, end, _, _ in ner_entries]):
                print(f'Head not found in NER: {relation["head"]}')
                problem_relations.append(relation)
            if not any([tail_start == start and tail_end == end for start, end, _, _ in ner_entries]):
                print(f'Tail not found in NER: {relation["tail"]}')
                problem_relations.append(relation)

        # Append to transformed data
        transformed_data.append({
            "ner": ner_entries,
            "relations": relations,
            "tokenized_text": tokens,
        })

    print(f'Problem entities: {len(problem_entities)}')
    print(f'Problem relations: {len(problem_relations)}')

    return transformed_data

def find_type_by_position(data, start, end):
    vertexSet = data['vertexSet']
    tokens = data.get('tokens', [])
    token_slice = " ".join(tokens[start:end])

    # try to find the entity using token positions
    for vertex in vertexSet:
        if len(vertex['tokenpositions']) > 0 and vertex['tokenpositions'][0] == start and vertex['tokenpositions'][-1] + 1 == end:
            return vertex['type']
        
    # If not found by positions, try to match by lexicalInput
    for vertex in vertexSet:
        if vertex.get('lexicalInput'):
            # Normalize spaces in both strings for better matching
            lexical_input_normalized = " ".join(vertex['lexicalInput'].split())
            token_slice_normalized = " ".join(token_slice.split())
            if lexical_input_normalized == token_slice_normalized:
                return vertex['type']

    print(f'No type for vertex : {vertex}')

    return 'Unknown'

transformed_data = transform_wiki_zsl(data)

# shuffle
random.shuffle(transformed_data)

with open('./wiki_zsl_all.jsonl', 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item) + '\n')

