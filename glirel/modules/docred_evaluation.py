import argparse
import torch
import json
from glirel import GLiREL

def load_test_set():
    with open('../../data/docred_test.jsonl', 'r') as f:
        test_set = [json.load(l) for l in f]

    return test_set


def load_model(checkpoint_dir):
    model = GLiREL.from_pretrained(checkpoint_dir)
    return model

def run_inference(test_set, model):
    model.base_config.fixed_relation_types = False
    results, micro_f1, macro_f1, preds = model.evaluate(
        test_set, 
        flat_ner=True, 
        threshold=0.5, 
        batch_size=16,
        relation_types=[],
        top_k=1
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

    # init wikidata relation to ID map
    relation2id = {}

    submission = []
    for example, pred in (test_set, preds):
        instance = {}
        instance['title'] = example['title']
        instance['h_idx'] = pred['head']['pos']
        instance['t_idx'] = pred['tail']['pos']
        instance['r'] = relation2id.get(pred['relation'], 'NA')
        evidence = None
        submission.append(instance)

    # save submission file
    with open('submission.json', 'w') as f:
        json.dump(submission, f)

    # run docred eval script
