import argparse
import torch
import json
from glirel import GLiREL
import torch
import os
import sys
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
    model.device = device
    return model

def run_inference(test_set, model):
    model.base_config.fixed_relation_types = False
    results, metric_dict, preds = model.evaluate(
        test_set, 
        flat_ner=True, 
        threshold=model.base_config.eval_threshold, 
        batch_size=8,
        relation_types=[],
        top_k=1,
        return_preds=True
    )
    micro_f1, micro_precision, micro_recall = metric_dict['micro_f1'], metric_dict['micro_precision'], metric_dict['micro_recall']
    macro_f1, macro_precision, macro_recall = metric_dict['macro_f1'], metric_dict['macro_precision'], metric_dict['macro_recall']
    return preds, metric_dict

def get_gold_coreference_clusters(test_set):
    entity_to_cluster_idx_list = []
    clusters_list = []
    for example in test_set:
        entity_to_cluster_idx = {}
        clusters = {}
        for ent in example['position2clusterid']:
            position = (ent[0][0], ent[0][1])
            cluster_id = ent[1]
            entity_to_cluster_idx[position] = cluster_id
            clusters.setdefault(cluster_id, []).append(position)
        entity_to_cluster_idx_list.append(entity_to_cluster_idx)
        clusters_list.append(list(clusters.values()))
    return clusters_list, entity_to_cluster_idx_list

        

def run_evaluation(ckpt_dir, use_gold_coref=False, use_auxiliary_coref=False, model=None):

    log_file = None

    # Check if ckpt_dir exists and redirect print statements to a file if it does
    if os.path.exists(ckpt_dir):
        log_file = os.path.join(ckpt_dir, 'evaluation_log.txt')
        sys.stdout = open(log_file, 'a+')


    # Load the test set
    test_set = load_test_set()
    # test_set = test_set[:10]

    head_tail2rel_list = []
    title2head_tail2rel = {}
    title2tokenized_text = {}
    for example in test_set:
        title2tokenized_text[example['title']] = example['tokenized_text']
        head_tail2rel = {}
        for rel in example['relations']:
            head_tail2rel[(rel['head']['h_idx'], rel['tail']['t_idx'])] = rel['relation_text']
        head_tail2rel_list.append(head_tail2rel)
        title2head_tail2rel[example['title']] = head_tail2rel

    # Load the model
    if model is None:
        model = load_model(ckpt_dir)
    model.eval()

    # Run inference on the test set
    if not os.path.exists('data/re-docred/res'):
        os.makedirs('data/re-docred/res')
    preds, metric_dict = run_inference(test_set, model)
    preds = preds[metric_dict['best_threshold']]
    with open(INTERMEDIATE_RESULTS_PATH, 'w') as f:
        json.dump(preds, f)
    print(f"Inference done! Results saved to {INTERMEDIATE_RESULTS_PATH}")

    # get coreference clusters
    coref_clusters = []
    if use_auxiliary_coref: 
        raise NotImplementedError("Not implemented yet!")
        # TODO run fast-coref or some other coref model
    elif use_gold_coref: 
        gold_coref_clusters_batch, gold_coref_entity_to_cluster_idx = get_gold_coreference_clusters(test_set)
        coref_clusters.append((gold_coref_clusters_batch, gold_coref_entity_to_cluster_idx, 'GOLD_COREF'))

    # predicted coreference clusters
    pred_coref_clusters_batch, pred_coref_entity_to_cluster_idx = utils.get_coreference_clusters(preds)
    coref_clusters.append((pred_coref_clusters_batch, pred_coref_entity_to_cluster_idx, 'PRED_COREF'))

    # entity_to_cluster_idx_list --> (128, 129): 0, (33, 34): 0, (144, 145): 0, (0, 6): 0, ...


    for (clusters_batch, entity_to_cluster_idx, coref_type) in coref_clusters:
        
        print(f"Evaluating using {coref_type} coreference clusters...")
        
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
        with open(SUBMISSION_PATH, 'w') as f:
            json.dump(submission, f)
        print(f"Submission file saved to {SUBMISSION_PATH}!")

        with open(SUBMISSION_PATH, 'r') as f:
            submission = json.load(f)

        # run docred eval script
        print("Running official_evaluate script...")
        best_f1, _, best_f1_ign, _, best_p, best_r, debug_results = official_evaluate(   # official_evaluate_benchmark
            tmp=submission, 
            path='data/re-docred', 
            train_file='data/train_revised.json', 
            dev_file='data/test_revised.json', 
        )
        print(f"Scores: F1: {best_f1}, F1 Ignore: {best_f1_ign} Precision: {best_p}, Recall: {best_r}")


        # Create a mapping from title to index
        title_to_index = {example['title']: idx for idx, example in enumerate(test_set)}
        batch_tokenized_text = [example['tokenized_text'] for example in test_set]

        # Process false positives
        print("\nProcessing false positives:")
        for x in debug_results['false_positives']:
            title, r, h_idx, t_idx = x['title'], x['r'], x['h_idx'], x['t_idx']

            idx = title_to_index.get(title)
            if idx is None:
                print(f"Title not found: {title}")
                continue
            clusters = clusters_batch[idx]
            tokenized_text = batch_tokenized_text[idx]
            if h_idx >= len(clusters) or t_idx >= len(clusters):
                print(f"Invalid h_idx or t_idx for title {title}")
                continue
            head_cluster_positions = clusters[h_idx]
            tail_cluster_positions = clusters[t_idx]
            head_cluster = [' '.join(tokenized_text[s:e]) for s, e in head_cluster_positions]
            tail_cluster = [' '.join(tokenized_text[s:e]) for s, e in tail_cluster_positions]
            pred_relation = id2rel.get(r, 'NA')
            if (title, h_idx, t_idx) in debug_results['actual']: 
                # there is an actual relation that was misclassified
                actual_relation = debug_results['actual'][(title, h_idx, t_idx)]
                actual_relation = id2rel[actual_relation]
            else:
                # there's no relation between head and tail
                actual_relation = 'NA'  
            print(f"##################### False Positive in: {title}\n")
            print(f"Predicted Relation: {pred_relation}")
            print(f"Actual Relation: {actual_relation}")
            print(f"Head Cluster: {head_cluster}")
            print(f"Tail Cluster: {tail_cluster}")

        # Process false negatives
        print("\nProcessing false negatives:")
        for title, r, h_idx, t_idx in debug_results['false_negatives']:

            idx = title_to_index.get(title)
            if idx is None:
                print(f"Title not found: {title}")
                continue
            clusters = clusters_batch[idx]
            tokenized_text = batch_tokenized_text[idx]
            if h_idx >= len(clusters) or t_idx >= len(clusters):
                print(f"Invalid h_idx or t_idx for title {title}")
                continue
            head_cluster_positions = clusters[h_idx]
            tail_cluster_positions = clusters[t_idx]
            head_cluster = [' '.join(tokenized_text[s:e]) for s, e in head_cluster_positions]
            tail_cluster = [' '.join(tokenized_text[s:e]) for s, e in tail_cluster_positions]
            actual_relation = id2rel.get(r, 'NA')
            print(f"##################### False Negative in: {title}\n")
            print(f"Actual Relation: {actual_relation}")
            print(f"Head Cluster: {head_cluster}")
            print(f"Tail Cluster: {tail_cluster}")


        print(f"Scores {coref_type}: F1: {best_f1}, F1 Ignore: {best_f1_ign} Precision: {best_p}, Recall: {best_r}")

    if log_file:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    return best_f1, best_f1_ign, best_p, best_r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, help="Path to the model checkpoint directory")
    parser.add_argument("--use-gold-coref", action='store_true', help="Use gold coreference clusters")
    parser.add_argument("--use-auxiliary-coref", action='store_true', help="Use auxiliary coreference clusters")
    args = parser.parse_args()
    run_evaluation(ckpt_dir=args.ckpt_dir, use_gold_coref=args.use_gold_coref, use_auxiliary_coref=args.use_auxiliary_coref)