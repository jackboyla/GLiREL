import argparse
import torch
import json
from glirel import GLiREL

"""
gdown --folder https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw
"""

def load_test_set():
    with open('../../data/docred_test.jsonl', 'r') as f:
        test_set = [json.loads(l) for l in f]

    return test_set


def load_model(checkpoint_dir):
    model = GLiREL.from_pretrained(checkpoint_dir).to('cuda')
    return model

def run_inference(test_set, model):
    model.base_config.fixed_relation_types = False
    results, micro_f1, macro_f1, preds = model.evaluate(
        test_set, 
        flat_ner=True, 
        threshold=0.000001, 
        batch_size=16,
        relation_types=[],
        top_k=1,
        return_preds=True
    )
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, help="Path to the model checkpoint directory")
    args = parser.parse_args()

    # Load the test set
    test_set = load_test_set()

    # Load the model
    model = load_model(args.checkpoint_dir)

    # Run inference on the test set
    preds = run_inference(test_set, model)
    print("Inference done!")

    # init wikidata relation to ID map
    with open('../../data/ref/rel_info.json', 'r') as f:
        pid2name = json.load(f)      
        relation2id = {v: k for k, v in pid2name.items()}


    submission = []
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
            instance['r'] = relation2id.get(rel['relation_text'], 'NA')
            instance['evidence'] = None
            submission.append(instance)
        
        idx += 1

    # save submission file
    save_path = '../../data/res/result.json'
    with open(save_path, 'w') as f:
        json.dump(submission, f)
    print(f"Submission file saved to {save_path}!")

    # run docred eval script
    import subprocess
    subprocess.run(['python', '../../data/evaluation_official_docred.py', '../../data/', '../../data/docred_scores/'])
