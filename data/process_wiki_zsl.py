import json
import random
import os
random.seed(12)

from tqdm import tqdm
import gdown

file_id = '1TMYvAbe9wsB5GiWcUL5bMAs9x6CpvnAj' # https://github.com/vhientran/Code-ZSRE?tab=readme-ov-file
output = 'wiki_all.json'
if not os.path.exists(output):
    print("Downloading Wiki_ZSL dataset...")
    gdown.download(id=file_id, output=output, quiet=False)


with open('wiki_all.json', 'r') as f:    # len --> 93483
    dataset = json.load(f)

with open('all_wikidata_properties.json', 'r') as f:
    properties = json.load(f)      
    pid2name = {}
    for property in properties:
        pid = property['property'].split('/')[-1]
        pid2name[pid] = [property['propertyLabel']]

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

    MAX_ENTITY_LENGTH = 50

    transformed_data = []
    problem_entities = []
    problem_relations = []

    for idx, data in enumerate(dataset):

        tokens = data['tokens']

        # Prepare to collect NER entries
        ner_entries = []
        # NOTE: sort by length (some entities are substrings of others)
        data['vertexSet'] = sorted(data['vertexSet'], key=lambda x: len(x['tokenpositions']))
        for vertex in data['vertexSet']:
            if vertex.get('tokenpositions'):
                start = vertex['tokenpositions'][0]
                end = vertex['tokenpositions'][-1]
                entity_text = " ".join(tokens[start:end + 1])
                ner_entries.append([start, end, vertex['type'], entity_text])
                # try:         # NOTE: often false alarm due to tokenization differences
                #     assert any([tok in entity_text.split(" ") for tok in vertex['lexicalInput'].split(" ")]) or \
                #         any([tok in vertex['lexicalInput'].split(" ") for tok in entity_text.split(" ")]) \
                #             or vertex['type'] == 'DATE'
                # except:
                #     problem = "entity text and lexical input do not match"
                #     import ipdb; ipdb.set_trace()
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


        # De-duplicate NER entries
        unique_texts = set([text for _, _, _, text in ner_entries])
        unique_positions = set([(start, end) for start, end, _, _ in ner_entries])
        de_duplicated_ner = []
        for start, end, type, text in ner_entries:
            if (start, end) in unique_positions:
                unique_positions.remove((start, end))
                de_duplicated_ner.append([start, end, type, text])
            else:
                # print(f"Duplicate entity for idx {idx}: {start, end, type, text}")  # NOTE: seems to be all DATEs that cause this problem
                problem_entities.append((start, end))

        ner_entries = de_duplicated_ner

        # check for long entities
        reasonable_len_ner = []
        for start, end, type, text in ner_entries:
            if len(text) > MAX_ENTITY_LENGTH:
                problem = f"Long entity: {text} (idx {idx}) with length {len(text)}. Removing..."
                print(problem)
                problem_entities.append((start, end))
            else:
                reasonable_len_ner.append([start, end, type, text])
        ner_entries = reasonable_len_ner

        # Assert that no NER indices overlap with each other
        non_overlapping_ner = []
        removed_span2preferred_span = {}
        for i, (start1, end1, type1, text1) in enumerate(ner_entries):
            keep = True
            for j, (start2, end2, type2, text2) in enumerate(ner_entries):
                if i != j:
                    if not (end1 < start2 or end2 < start1):  # There is an overlap
                        if len(text1) < len(text2):
                            print(f"Removing {text1} in favor of {text2} due to overlap (idx {idx})")
                            keep = False
                            removed_span2preferred_span[(start1, end1)] = (start2, end2)
                            break

            if keep:
                non_overlapping_ner.append((start1, end1, type1, text1))

        ner_entries = non_overlapping_ner
        

        # Prepare to collect relations
        relations = []
        for edge in data['edgeSet']:
            if len(edge['left']) == 0 or len(edge['right']) == 0: # Skip relations with no positions
                problem_relations.append(edge)
                continue
            if edge['kbID'] not in pid2name:   # Skip relations not found in pid2name
                print('Relation not found:', edge['kbID'])
                continue
            head_start = edge['left'][0]
            head_end = edge['left'][-1]
            tail_start = edge['right'][0]
            tail_end = edge['right'][-1]
            if (head_start, head_end) in removed_span2preferred_span:
                head_start, head_end = removed_span2preferred_span[(head_start, head_end)]
            if (tail_start, tail_end) in removed_span2preferred_span:
                tail_start, tail_end = removed_span2preferred_span[(tail_start, tail_end)]
            head_text = " ".join(tokens[head_start:head_end + 1]) # +1 to include last token
            tail_text = " ".join(tokens[tail_start:tail_end + 1])
            if len(head_text) > MAX_ENTITY_LENGTH:
                print(f"Skipping relation with long head: {head_text}")
                problem_relations.append(edge)
                continue
            if len(tail_text) > MAX_ENTITY_LENGTH:
                print(f"Skipping relation with long tail: {tail_text}")
                problem_relations.append(edge)
                continue

            try:
                assert head_text in unique_texts, f"Head not found in NER: {head_text}"
                assert tail_text in unique_texts, f"Tail not found in NER: {tail_text}"
            except:
                problem = "relation head or tail not found in NER"
                import ipdb; ipdb.set_trace()

            # Find and append relation
            if len(pid2name[edge['kbID']][0]) < 2:
                import ipdb; ipdb.set_trace()
            
            relations.append({
                "head": {"mention": head_text, "position": [head_start, head_end], "type": find_type_by_position(data, head_start, head_end)}, # +1 to include last token
                "tail": {"mention": tail_text, "position": [tail_start, tail_end], "type": find_type_by_position(data, tail_start, tail_end)},
                "relation_id": edge['kbID'],
                "relation_text": pid2name[edge['kbID']][0],
            })

        # ensure that all positions in relations exist in ner
        for relation in relations:
            head_start, head_end = relation['head']['position'][0], relation['head']['position'][1]
            tail_start, tail_end = relation['tail']['position'][0], relation['tail']['position'][1]
            if not any([head_start == start and head_end == end for start, end, _, _ in ner_entries]):
                problem = f'Head not found in NER: {relation["head"]}'
                print(problem)
                import ipdb; ipdb.set_trace()
                problem_relations.append(relation)
            if not any([tail_start == start and tail_end == end for start, end, _, _ in ner_entries]):
                problem = f'Tail not found in NER: {relation["tail"]}'
                print(problem)
                import ipdb; ipdb.set_trace()
                problem_relations.append(relation)

        # Ensure that no relation has duplicate positions
        de_duplicated_relation = []
        relation_pos = set()
        for relation in relations:
            position_tuple = (tuple(relation['head']['position']), tuple(relation['tail']['position']))
            if position_tuple in relation_pos:
                print(f"Duplicate relation: {relation}")
                problem_relations.append(relation)
            else:
                relation_pos.add(position_tuple)
                de_duplicated_relation.append(relation)
        relations = de_duplicated_relation


        if len(relations) > 0:
            # Append to transformed data
            transformed_data.append({
                "ner": ner_entries,
                "relations": relations,
                "tokenized_text": tokens,
            })
        else:
            print(f"Skipping idx {idx} due to no relations")

    print(f'Problem entities: {len(problem_entities)}')
    print(f'Problem relations: {len(problem_relations)}')

    """
    Problem entities: 862
    Problem relations: 199
    """

    return transformed_data

def find_type_by_position(data, start, end):
    vertexSet = data['vertexSet']
    tokens = data.get('tokens', [])
    token_slice = " ".join(tokens[start:end + 1])  # Include the last token

    # Try to find the entity using token positions
    for vertex in vertexSet:
        if len(vertex['tokenpositions']) > 0 and vertex['tokenpositions'][0] == start and vertex['tokenpositions'][-1] == end:
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
# random.seed(12)
random.shuffle(transformed_data)

# count up relation types
relationship_counts = {}
for item in tqdm(transformed_data, desc="Counting relations"):
    relations = item['relations']
    for relation in relations:
        relation_text = relation['relation_text']
        if relation_text in relationship_counts:
            relationship_counts[relation_text] += 1
        else:
            relationship_counts[relation_text] = 1

print(f"Relationship counts: {relationship_counts}")
with open(f"wikiw_zsl_type_counts.json", "w") as f:
    f.write(json.dumps(relationship_counts))


save_path = './wiki_zsl_all.jsonl'
with open(save_path, 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item) + '\n')
print(f"Saved to {save_path}")
