from datasets import load_dataset
import json, statistics
from tqdm import tqdm
from random import randint

'''

ReDocRED paper: https://arxiv.org/pdf/2205.12696.pdf
dataset: https://github.com/tonytan48/Re-DocRED/tree/main

all entities in DocRED are grouped together by co-reference resolution, 
and each group is treated as a single entity.

We want to split that up so that each mention is a separate entity.
This involves duplicating the relationships shared across the mentions of the same entity.
As well as creating a `SELF` relationship for each mention to mentions of the same entity.

Also getting the rest of the data in the right format for GLiREL training (ner, tokenized_text)

'''

with open('all_wikidata_properties.json', 'r') as f:
    properties = json.load(f)      
    id2rel = {}
    for property in properties:
        pid = property['property'].split('/')[-1]
        id2rel[pid] = [property['propertyLabel']]

id2rel.update(json.load(open('pid2name_fewrel.json')))
id2rel.update(json.load(open('pid2name_wiki.json')))
# NEW / CHANGED PROPERTIES
# P7 (brother) --> P3373 (sibling)
id2rel['P7'] = ["sibling", "the subject and the object have at least one common parent (brother, sister, etc. including half-siblings)"]
# P9 (sister) --> P3373 (sibling)
id2rel['P9'] = ["sibling", "the subject and the object have at least one common parent (brother, sister, etc. including half-siblings)"]

id2rel = {key: value[0] for key, value in id2rel.items()}
rel2id = {value: key for key, value in id2rel.items()}

duplicate_entities = []
duplicate_relations = []
overlapping_entities = []

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


def expand_labels(doc, idx):
    expanded_labels = []
    
    # Function to get cumulative position of an entity mention
    def get_cumulative_position(mention, cumulative_sentence_lengths):
        return cumulative_sentence_lengths[mention['sent_id']] + mention['pos'][0], \
               cumulative_sentence_lengths[mention['sent_id']] + mention['pos'][1]
    
    # Calculate cumulative sentence lengths for indexing
    cumulative_sentence_lengths = [0]
    for sentence in doc['sents']:
        cumulative_sentence_lengths.append(cumulative_sentence_lengths[-1] + len(sentence))

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
    
    # Iterate over the relations in the labels
    for i, relation in enumerate(doc['labels']):

        head_id, tail_id, relation_id, evidence = relation['h'], relation['t'], relation['r'], relation['evidence']
        relation_text = id2rel[relation_id]
        
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

    
    # De-duplicate relations
    expanded_labels = sorted(expanded_labels, key=lambda x: (x['head']['name'], x['head']['position'][0], x['tail']['position'][0]))
    relation_pos = {}
    de_duplicated_expanded_labels = []
    for r in expanded_labels:
        position_tuple = (tuple(r['head']['position']), tuple(r['tail']['position']))
        if position_tuple in relation_pos: 
            print(f"Duplicate position for relation in (idx {idx}) Relation {r} \n--already as-->\n {relation_pos[position_tuple]}\n")
            duplicate_relations.append(r)
        else:
            de_duplicated_expanded_labels.append(r)
            relation_pos[position_tuple] = r
    
    expanded_labels = de_duplicated_expanded_labels

    # sort for viewing
    expanded_labels = sorted(expanded_labels, key=lambda x: (x['head']['name'], x['head']['position'][0], x['tail']['position'][0]))
    return expanded_labels


def get_ner_from_entity_mapping(entity_mapping, doc, idx):
    ner_entries = []
    position2clusterid = []
    cumulative_offset = 0  
    for k, mention in entity_mapping.items():
        # Find the cumulative offset for the mention (add up the len of all sentences before the mention's sentence)
        cumulative_offset = sum(len(sent) for sent in doc['sents'][:mention['sent_id']])
        # Adjust the entity's position with the cumulative offset
        start = cumulative_offset + mention['pos'][0]
        end = cumulative_offset + mention['pos'][1]
        ner_entries.append([start, end, mention['type'], mention['name']])

        position2clusterid.append([[start, end], k[0]])

    # De-duplicate NER entries
    unique_texts = set([text for _, _, _, text in ner_entries])
    unique_positions = set([(start, end) for start, end, _, _ in ner_entries])
    de_duplicated_ner = []
    for start, end, type, text in ner_entries:
        if (start, end) in unique_positions:
            unique_positions.remove((start, end))
            de_duplicated_ner.append([start, end, type, text])
        else:
            print(f"Duplicate entity for idx {idx}: {start, end, type, text}")  # NOTE: seems to be all DATEs that cause this problem
            duplicate_entities.append((start, end))

    ner_entries = de_duplicated_ner

    # Assert that no NER indices overlap with each other
    non_overlapping_ner = []
    removed_span2preferred_span = {}
    for i, (start1, end1, type1, text1) in enumerate(ner_entries):
        keep = True
        for j, (start2, end2, type2, text2) in enumerate(ner_entries):
            if i != j:
                if not (end1 < start2 or end2 < start1):  # There is an overlap
                    overlapping_entities.append((start1, end1, type1, text1))
                    if len(text1) < len(text2):
                        print(f"Removing {text1} in favor of {text2} due to overlap (idx {idx})")
                        keep = False
                        removed_span2preferred_span[(start1, end1)] = (start2, end2)
                        break

        if keep:
            non_overlapping_ner.append((start1, end1, type1, text1))

    ner_entries = non_overlapping_ner

    ner_entries = sorted(ner_entries, key=lambda x: x[0])
    return ner_entries, position2clusterid


def assert_ner_aligns_with_text(ner, tokenized_text, i):
    for start, end, _, ner_surface_form in ner:

        target_start, target_end = ner_surface_form.split(' ')[0], ner_surface_form.split(' ')[-1]
        text_surface_form = tokenized_text[start:end]
        start, end = text_surface_form[0], text_surface_form[-1]
        assert target_start == start and target_end == end, f"Error in document {i} --> surface_form: {ner_surface_form} != Text: {text_surface_form}"


for SPLIT in ['train', 'dev', 'test']: # train_distant
    data = []
    doc_lens = []
    with open(f're-docred/data/{SPLIT}_revised.json', 'r') as f:
        dataset = json.load(f)
    for i in tqdm(range(len(dataset))):
        doc = dataset[i]
        doc_lens.append(sum(len(s) for s in doc['sents']))
        example_row = {'title': doc['title']}

        # Combine the list of sentences into one list of tokens
        example_row['tokenized_text'] = [token for sentence in doc['sents'] for token in sentence]

        entity_mapping = map_entities(doc)

        example_row['ner'], position2clusterid = get_ner_from_entity_mapping(entity_mapping, doc, i)  #  "ner": [[3,6,"LOC"], [7,9,"PER"]]
        example_row['position2clusterid'] = position2clusterid
        # assert_ner_aligns_with_text(example_row['ner'], example_row['tokenized_text'], i)
        example_row['relations'] = expand_labels(doc, i)

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
    # are all title keys unique?
    assert len(data) == len(set([doc['title'] for doc in data]))
    print(f"All unique titles")

    print(f"Duplicate entities: {len(duplicate_entities)}")
    print(f"Overlapping entities: {len(overlapping_entities)}")
    print(f"Duplicate relations: {len(duplicate_relations)}")

    data = sorted(data, key=lambda x: len(x['ner']), reverse=True)

    # save to a jsonl file
    with open(f'redocred_{SPLIT}.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")





'''
Aspren [0, 1] -> SELF -> Asprenas [2, 3]
Aspren [0, 1] -> religion -> Christian [10, 11]
Aspren [0, 1] -> SELF -> Aspren [21, 22]
Aspren [0, 1] -> SELF -> Aspren [54, 55]
Aspren [0, 1] -> SELF -> Aspren [94, 95]
Aspren [0, 1] -> SELF -> Aspren [111, 112]
Aspren [0, 1] -> religion -> Christianity [172, 173]
Aspren [0, 1] -> SELF -> Aspren [182, 183]
Aspren [0, 1] -> SELF -> Aspren [206, 207]
Aspren [21, 22] -> SELF -> Aspren [0, 1]
Aspren [21, 22] -> SELF -> Asprenas [2, 3]
Aspren [21, 22] -> religion -> Christian [10, 11]
Aspren [21, 22] -> SELF -> Aspren [54, 55]
Aspren [21, 22] -> SELF -> Aspren [94, 95]
Aspren [21, 22] -> SELF -> Aspren [111, 112]
Aspren [21, 22] -> religion -> Christianity [172, 173]
Aspren [21, 22] -> SELF -> Aspren [182, 183]
Aspren [21, 22] -> SELF -> Aspren [206, 207]
Aspren [54, 55] -> SELF -> Aspren [0, 1]
Aspren [54, 55] -> SELF -> Asprenas [2, 3]
Aspren [54, 55] -> religion -> Christian [10, 11]
Aspren [54, 55] -> SELF -> Aspren [21, 22]
Aspren [54, 55] -> SELF -> Aspren [94, 95]
Aspren [54, 55] -> SELF -> Aspren [111, 112]
Aspren [54, 55] -> religion -> Christianity [172, 173]
Aspren [54, 55] -> SELF -> Aspren [182, 183]
Aspren [54, 55] -> SELF -> Aspren [206, 207]
Aspren [94, 95] -> SELF -> Aspren [0, 1]
Aspren [94, 95] -> SELF -> Asprenas [2, 3]
Aspren [94, 95] -> religion -> Christian [10, 11]
Aspren [94, 95] -> SELF -> Aspren [21, 22]
Aspren [94, 95] -> SELF -> Aspren [54, 55]
Aspren [94, 95] -> SELF -> Aspren [111, 112]
Aspren [94, 95] -> religion -> Christianity [172, 173]
Aspren [94, 95] -> SELF -> Aspren [182, 183]
Aspren [94, 95] -> SELF -> Aspren [206, 207]
Aspren [111, 112] -> SELF -> Aspren [0, 1]
Aspren [111, 112] -> SELF -> Asprenas [2, 3]
Aspren [111, 112] -> religion -> Christian [10, 11]
Aspren [111, 112] -> SELF -> Aspren [21, 22]
Aspren [111, 112] -> SELF -> Aspren [54, 55]
Aspren [111, 112] -> SELF -> Aspren [94, 95]
Aspren [111, 112] -> religion -> Christianity [172, 173]
Aspren [111, 112] -> SELF -> Aspren [182, 183]
Aspren [111, 112] -> SELF -> Aspren [206, 207]
Aspren [182, 183] -> SELF -> Aspren [0, 1]
Aspren [182, 183] -> SELF -> Asprenas [2, 3]
Aspren [182, 183] -> religion -> Christian [10, 11]
Aspren [182, 183] -> SELF -> Aspren [21, 22]
Aspren [182, 183] -> SELF -> Aspren [54, 55]
Aspren [182, 183] -> SELF -> Aspren [94, 95]
Aspren [182, 183] -> SELF -> Aspren [111, 112]
Aspren [182, 183] -> religion -> Christianity [172, 173]
Aspren [182, 183] -> SELF -> Aspren [206, 207]
Aspren [206, 207] -> SELF -> Aspren [0, 1]
Aspren [206, 207] -> SELF -> Asprenas [2, 3]
Aspren [206, 207] -> religion -> Christian [10, 11]
Aspren [206, 207] -> SELF -> Aspren [21, 22]
Aspren [206, 207] -> SELF -> Aspren [54, 55]
Aspren [206, 207] -> SELF -> Aspren [94, 95]
Aspren [206, 207] -> SELF -> Aspren [111, 112]
Aspren [206, 207] -> religion -> Christianity [172, 173]
Aspren [206, 207] -> SELF -> Aspren [182, 183]
Asprenas [2, 3] -> SELF -> Aspren [0, 1]
Asprenas [2, 3] -> religion -> Christian [10, 11]
Asprenas [2, 3] -> SELF -> Aspren [21, 22]
Asprenas [2, 3] -> SELF -> Aspren [54, 55]
Asprenas [2, 3] -> SELF -> Aspren [94, 95]
Asprenas [2, 3] -> SELF -> Aspren [111, 112]
Asprenas [2, 3] -> religion -> Christianity [172, 173]
Asprenas [2, 3] -> SELF -> Aspren [182, 183]
Asprenas [2, 3] -> SELF -> Aspren [206, 207]
Calendario Marmoreo di Napoli [87, 91] -> SELF -> The Marble Calendar of Naples [81, 86]
Candida [191, 192] -> SELF -> Candida the Elder [156, 159]
Candida [191, 192] -> religion -> Christianity [172, 173]
Candida the Elder [156, 159] -> religion -> Christianity [172, 173]
Candida the Elder [156, 159] -> SELF -> Candida [191, 192]
Naples [19, 20] -> country -> Roman Republic [65, 67]
Naples [19, 20] -> country -> Roman Empire [73, 75]
Naples [19, 20] -> located in the administrative territorial entity -> Roman Empire [73, 75]
Naples [19, 20] -> SELF -> Naples [147, 148]
Naples [19, 20] -> SELF -> Naples [179, 180]
Naples [19, 20] -> SELF -> Naples [210, 211]
Naples [147, 148] -> SELF -> Naples [19, 20]
Naples [147, 148] -> country -> Roman Republic [65, 67]
Naples [147, 148] -> country -> Roman Empire [73, 75]
Naples [147, 148] -> located in the administrative territorial entity -> Roman Empire [73, 75]
Naples [147, 148] -> SELF -> Naples [179, 180]
Naples [147, 148] -> SELF -> Naples [210, 211]
Naples [179, 180] -> SELF -> Naples [19, 20]
Naples [179, 180] -> country -> Roman Republic [65, 67]
Naples [179, 180] -> country -> Roman Empire [73, 75]
Naples [179, 180] -> located in the administrative territorial entity -> Roman Empire [73, 75]
Naples [179, 180] -> SELF -> Naples [147, 148]
Naples [179, 180] -> SELF -> Naples [210, 211]
Naples [210, 211] -> SELF -> Naples [19, 20]
Naples [210, 211] -> country -> Roman Republic [65, 67]
Naples [210, 211] -> country -> Roman Empire [73, 75]
Naples [210, 211] -> located in the administrative territorial entity -> Roman Empire [73, 75]
Naples [210, 211] -> SELF -> Naples [147, 148]
Naples [210, 211] -> SELF -> Naples [179, 180]
Neapolitan Church [45, 47] -> country -> Roman Republic [65, 67]
Peter [189, 190] -> religion -> Christian [10, 11]
Peter [189, 190] -> work location -> Rome [143, 144]
Peter [189, 190] -> religion -> Christianity [172, 173]
Peter [189, 190] -> SELF -> Peter [204, 205]
Peter [204, 205] -> religion -> Christian [10, 11]
Peter [204, 205] -> work location -> Rome [143, 144]
Peter [204, 205] -> religion -> Christianity [172, 173]
Peter [204, 205] -> SELF -> Peter [189, 190]
Roman Empire [73, 75] -> follows -> Roman Republic [65, 67]
Roman Republic [65, 67] -> followed by -> Roman Empire [73, 75]
Saint Peter [136, 138] -> religion -> Christian [10, 11]
Saint Peter [136, 138] -> residence -> Rome [143, 144]
Saint Peter [136, 138] -> work location -> Rome [143, 144]
Saint Peter [136, 138] -> religion -> Christianity [172, 173]
San Pietro ad Aram [236, 240] -> located in the administrative territorial entity -> Naples [19, 20]
San Pietro ad Aram [236, 240] -> located in the administrative territorial entity -> Roman Empire [73, 75]
San Pietro ad Aram [236, 240] -> located in the administrative territorial entity -> Naples [147, 148]
San Pietro ad Aram [236, 240] -> located in the administrative territorial entity -> Naples [179, 180]
San Pietro ad Aram [236, 240] -> located in the administrative territorial entity -> Naples [210, 211]
Santa Maria del Principio [219, 223] -> located in the administrative territorial entity -> Naples [19, 20]
Santa Maria del Principio [219, 223] -> located in the administrative territorial entity -> Roman Empire [73, 75]
Santa Maria del Principio [219, 223] -> located in the administrative territorial entity -> Naples [147, 148]
Santa Maria del Principio [219, 223] -> located in the administrative territorial entity -> Naples [179, 180]
Santa Maria del Principio [219, 223] -> located in the administrative territorial entity -> Naples [210, 211]
Santa Restituta [233, 235] -> located in the administrative territorial entity -> Naples [19, 20]
Santa Restituta [233, 235] -> located in the administrative territorial entity -> Roman Empire [73, 75]
Santa Restituta [233, 235] -> located in the administrative territorial entity -> Naples [147, 148]
Santa Restituta [233, 235] -> located in the administrative territorial entity -> Naples [179, 180]
Santa Restituta [233, 235] -> located in the administrative territorial entity -> Naples [210, 211]
The Marble Calendar of Naples [81, 86] -> SELF -> Calendario Marmoreo di Napoli [87, 91]
Number of documents: 500
Median length of documents: 182.0
Max length of documents: 481
Median number of relations per document: 104.0
Max number of relations per document: 754
Median number of spans per document: 26.0
Max number of spans per document: 62
All unique titles
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 671.15it/s]
Antarctic Peninsula [177, 179] -> located in or next to body of water -> Southern Ocean [17, 19]
Antarctic Peninsula [177, 179] -> continent -> Antarctica [40, 41]
Antarctic Peninsula [177, 179] -> located in or next to body of water -> Scotia Sea [153, 155]
Drake Passage [33, 35] -> continent -> Antarctica [40, 41]
Patagonia [174, 175] -> located in or next to body of water -> South Atlantic [14, 16]
Patagonia [174, 175] -> located in or next to body of water -> Southern Ocean [17, 19]
Scotia Plate [1, 3] -> located on terrain feature -> Southern Ocean [17, 19]
Scotia Plate [1, 3] -> SELF -> Scotia Plate [139, 141]
Scotia Plate [139, 141] -> SELF -> Scotia Plate [1, 3]
Scotia Plate [139, 141] -> located on terrain feature -> Southern Ocean [17, 19]
Scotia Sea [153, 155] -> part of -> Southern Ocean [17, 19]
Scottish National Antarctic Expedition [116, 120] -> start time -> 1902 [121, 122]
South America [37, 39] -> located in or next to body of water -> South Atlantic [14, 16]
South America [37, 39] -> located in or next to body of water -> Southern Ocean [17, 19]
South America [37, 39] -> located in or next to body of water -> Scotia Sea [153, 155]
South America [37, 39] -> SELF -> South America [213, 215]
South America [213, 215] -> located in or next to body of water -> South Atlantic [14, 16]
South America [213, 215] -> located in or next to body of water -> Southern Ocean [17, 19]
South America [213, 215] -> SELF -> South America [37, 39]
South America [213, 215] -> located in or next to body of water -> Scotia Sea [153, 155]
South American plate [62, 65] -> located in or next to body of water -> South Atlantic [14, 16]
South Georgia Islands [201, 204] -> located in or next to body of water -> South Atlantic [14, 16]
South Georgia Islands [201, 204] -> located in or next to body of water -> Southern Ocean [17, 19]
South Georgia Islands [201, 204] -> located in or next to body of water -> Scotia Sea [153, 155]
Southern Ocean [17, 19] -> has part -> Scotia Sea [153, 155]
Number of documents: 500
Median length of documents: 180.0
Max length of documents: 510
Median number of relations per document: 96.0
Max number of relations per document: 1479
Median number of spans per document: 25.0
Max number of spans per document: 57
All unique titles
'''