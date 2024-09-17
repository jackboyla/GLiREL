import argparse
import torch
import json
from glirel import GLiREL
import torch
import os
from evaluation import official_evaluate

"""
https://github.com/tonytan48/Re-DocRED/tree/main/data

python data/re-docred/run_evaluation.py \
    --ckpt-dir logs/docred/docred-2024-09-16__13-06-14/model_19500 \
    --use-gold-coref

"""

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, help="Path to the model checkpoint directory")
    parser.add_argument("--use-gold-coref", action='store_true', help="Use gold coreference clusters")
    args = parser.parse_args()

    # Load the test set
    test_set = load_test_set()
    test_set = test_set[:10]

    # Load the model
    model = load_model(args.ckpt_dir)

    # Run inference on the test set
    preds = run_inference(test_set, model)
    print("Inference done!")

    submission = []
    if args.use_gold_coref: 
        idx = 0
        for example, pred in zip(test_set, preds):
            ner_pos2clusterid = {}
            for i in example['position2clusterid']:
                ner_pos2clusterid[(i[0][0], i[0][1])] = i[1]
            for rel in example['relations']:
                head_pos = rel['head']['position']
                tail_pos = rel['tail']['position']
                ner_pos2clusterid[(head_pos[0], head_pos[1])] = rel['head']['h_idx']
                ner_pos2clusterid[(tail_pos[0], tail_pos[1])] = rel['tail']['t_idx']
            
            for rel in pred:
                instance = {}
                instance['title'] = example['title']
                head_pos = rel['head']['position']
                tail_pos = rel['tail']['position']
                try:
                    instance['h_idx'] = ner_pos2clusterid[(head_pos[0], head_pos[1])]
                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()
                try:
                    instance['t_idx'] = ner_pos2clusterid[(tail_pos[0], tail_pos[1])]
                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()
                instance['r'] = rel2id.get(rel['relation_text'], 'NA')
                instance['evidence'] = None
                submission.append(instance)
            idx += 1
    else:
        raise NotImplementedError("Not implemented yet!")
        

    # save submission file
    if not os.path.exists('data/re-docred/res'):
        os.makedirs('data/re-docred/res')
    save_path = 'data/re-docred/res/result.json'
    with open(save_path, 'w') as f:
        json.dump(submission, f)
    print(f"Submission file saved to {save_path}!")

    # run docred eval script
    best_f1, _, best_f1_ign, _, best_p, best_r = official_evaluate(
        tmp=submission, 
        path='data/re-docred', 
        train_file='data/train_revised.json', 
        dev_file='data/train_revised.json', 
    )
    print(f"Scores: F1: {best_f1}, F1 Ignore: {best_f1_ign} Precision: {best_p}, Recall: {best_r}")
