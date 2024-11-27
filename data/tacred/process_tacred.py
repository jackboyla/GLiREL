import json

with open('./Few_Shot_transformation_and_sampling/categories_split.json') as f:
    categories_split = json.load(f)
    # [('train', 26), ('dev', 7), ('test', 11)]

with open('./Few_Shot_transformation_and_sampling/data_few_shot/_dev_data.json') as f:
    dev_data = json.load(f)


print(f"DEV DATA")
replaced_rels_dev = set()
for item in dev_data['no_relation']:
    replaced_rels_dev.add(item['relation'])
print(f"Replaced relations: {replaced_rels_dev}")



with open('./Few_Shot_transformation_and_sampling/data_few_shot/_test_data.json') as f:
    test_data = json.load(f)


print(f"TEST DATA")
replaced_rels_test = set()
for item in test_data['no_relation']:
    replaced_rels_test.add(item['relation'])
print(f"Replaced relations: {replaced_rels_test}")


# replaced_rels_test.intersection(replaced_rels_dev)

import ipdb; ipdb.set_trace()

'''
data.keys()
dict_keys(['no_relation', 'org:country_of_headquarters', 'per:age', 'org:parents', 'org:founded', 'per:stateorprovince_of_death', 'per:alternate_names'])


data['no_relation'][0]
{
    'id': 'e7798fb926b9403cfcd2', 
    'docid': 'APW_ENG_20101103.0539', 
    'relation': 'per:title', 
    'token': ['At', 'the', 'same', 'time', ',', 'Chief', 'Financial', 'Officer', 'Douglas', 'Flint', 'will', 'become', 'chairman', ',', 'succeeding', 'Stephen', 'Green', 'who', 'is', 'leaving', 'to', 'take', 'a', 'government', 'job', '.'], 
    'subj_start': 8, 
    'subj_end': 9, 
    'obj_start': 12, 
    'obj_end': 12, 
    'subj_type': 'PERSON', 
    'obj_type': 'TITLE', 
    'stanford_pos': ['IN', 'DT', 'JJ', 'NN', ',', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'MD', 'VB', 'NN', ',', 'VBG', 'NNP', 'NNP', 'WP', 'VBZ', 'VBG', 'TO', 'VB', 'DT', 'NN', 'NN', '.'], 
    'stanford_ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
    'stanford_head': [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12], 
    'stanford_deprel': ['case', 'det', 'amod', 'nmod', 'punct', 'compound', 'compound', 'compound', 'compound', 'nsubj', 'aux', 'ROOT', 'xcomp', 'punct', 'xcomp', 'compound', 'dobj', 'nsubj', 'aux', 'acl:relcl', 'mark', 'xcomp', 'det', 'compound', 'dobj', 'punct'], 
    'tokens': ['At', 'the', 'same', 'time', ',', 'Chief', 'Financial', 'Officer', 'Douglas', 'Flint', 'will', 'become', 'chairman', ',', 'succeeding', 'Stephen', 'Green', 'who', 'is', 'leaving', 'to', 'take', 'a', 'government', 'job', '.'], 
    'h': ['douglas flint', None, [[8, 9]]], 
    't': ['chairman', None, [[12]]]}
'''



