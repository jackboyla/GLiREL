import argparse
import torch
import json
from glirel import GLiREL
import torch
import os
from evaluation import official_evaluate
from glirel.modules import utils

"""
https://github.com/tonytan48/Re-DocRED/tree/main/data

python data/re-docred/run_evaluation.py \
    --ckpt-dir logs/redocred/redocred-2024-09-16__21-11-24/model_43800 \
    --use-gold-coref

"""

SUBMISSION_PATH = 'data/re-docred/res/result.json'
INTERMEDIATE_RESULTS_PATH = 'data/re-docred/res/intermediate_results.json'

with open('data/all_wikidata_properties.json', 'r') as f:
    properties = json.load(f)      
    id2rel = {}
    for property in properties:
        pid = property['property'].split('/')[-1]
        id2rel[pid] = [property['propertyLabel']]

id2rel.update(json.load(open('data/pid2name_fewrel.json')))
id2rel.update(json.load(open('data/pid2name_wiki.json')))
# NEW / CHANGED PROPERTIES
# P7 (brother) --> P3373 (sibling)
id2rel['P7'] = ["sibling", "the subject and the object have at least one common parent (brother, sister, etc. including half-siblings)"]
# P9 (sister) --> P3373 (sibling)
id2rel['P9'] = ["sibling", "the subject and the object have at least one common parent (brother, sister, etc. including half-siblings)"]
id2rel = {key: value[0] for key, value in id2rel.items()}
rel2id = {value: key for key, value in id2rel.items()}


def load_test_set():
    with open('data/redocred_test.jsonl', 'r') as f:
        test_set = [json.loads(l) for l in f]

    return test_set


def load_model(checkpoint_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GLiREL.from_pretrained(checkpoint_dir).to(device)
    return model

def run_inference(test_set, model):
    model.base_config.fixed_relation_types = False
    results, micro_f1, macro_f1, preds = model.evaluate(
        test_set, 
        flat_ner=True, 
        threshold=model.base_config.eval_threshold, 
        batch_size=16,
        relation_types=[],
        top_k=1,
        return_preds=True
    )
    return preds

def get_gold_coreference_clusters(test_set):
    entity_to_cluster_idx_list = []
    for example in test_set:
        entity_to_cluster_idx = {}
        for ent in example['position2clusterid']:
            entity_to_cluster_idx[(ent[0][0], ent[0][1])] = ent[1]
        entity_to_cluster_idx_list.append(entity_to_cluster_idx)
    return entity_to_cluster_idx_list
        

def run_evaluation(ckpt_dir, use_gold_coref=False, use_auxiliary_coref=False, model=None):

    # Load the test set
    test_set = load_test_set()

    # Load the model
    if model is None:
        model = load_model(ckpt_dir)
    model.eval()

    # Run inference on the test set
    preds = run_inference(test_set, model)
    with open(INTERMEDIATE_RESULTS_PATH, 'w') as f:
        json.dump(preds, f)
    print(f"Inference done! Results saved to {INTERMEDIATE_RESULTS_PATH}")

    # get coreference clusters
    if use_auxiliary_coref: 
        raise NotImplementedError("Not implemented yet!")
        # TODO run fast-coref or some other coref model
    elif use_gold_coref: 
        entity_to_cluster_idx = get_gold_coreference_clusters(test_set)
    else:
        # use predicted coreference clusters
        print("Using predicted coreference clusters...")
        clusters, entity_to_cluster_idx = utils.get_coreference_clusters(preds)

    # entity_to_cluster_idx_list --> (128, 129): 0, (33, 34): 0, (144, 145): 0, (0, 6): 0, ...

    # propagate labels using coreference clusters
    cluster_relations_list = utils.aggregate_cluster_relations(entity_to_cluster_idx, preds)
    
    # [ [{'h_idx': 0, 't_idx': 2, 'r': 'performer'}, ...], ...]

    submission = []
    for example, cluster_relations in zip(test_set, cluster_relations_list):
        for rel in cluster_relations:
            instance = {}
            instance['title'] = example['title']
            instance['h_idx'] = rel['h_idx']
            instance['t_idx'] = rel['t_idx']
            instance['r'] = rel2id.get(rel['r'], 'NA')
            instance['evidence'] = None
            submission.append(instance)


    '''
    submission: [
    {"title": "Loud Tour", "h_idx": 0, "t_idx": 2, "r": "P175", "evidence": null}, 
    {"title": "Loud Tour", "h_idx": 0, "t_idx": 2, "r": "P175", "evidence": null}, 
    {"title": "Loud Tour", "h_idx": 0, "t_idx": 2, "r": "P175", "evidence": null},
    ... 
    '''
        

    # save submission file
    if not os.path.exists('data/re-docred/res'):
        os.makedirs('data/re-docred/res')
    with open(SUBMISSION_PATH, 'w') as f:
        json.dump(submission, f)
    print(f"Submission file saved to {SUBMISSION_PATH}!")

    with open(SUBMISSION_PATH, 'r') as f:
        submission = json.load(f)

    # run docred eval script
    best_f1, _, best_f1_ign, _, best_p, best_r = official_evaluate(
        tmp=submission, 
        path='data/re-docred', 
        train_file='data/train_revised.json', 
        dev_file='data/test_revised.json', 
    )
    print(f"Scores: F1: {best_f1}, F1 Ignore: {best_f1_ign} Precision: {best_p}, Recall: {best_r}")
    return best_f1, best_f1_ign, best_p, best_r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, help="Path to the model checkpoint directory")
    parser.add_argument("--use-gold-coref", action='store_true', help="Use gold coreference clusters")
    parser.add_argument("--use-auxiliary-coref", action='store_true', help="Use auxiliary coreference clusters")
    args = parser.parse_args()
    run_evaluation(ckpt_dir=args.ckpt_dir, use_gold_coref=args.use_gold_coref, use_auxiliary_coref=args.use_auxiliary_coref)