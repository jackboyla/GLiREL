import argparse
import os

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

# from model_nested import NerFilteredSemiCRF
from glirel import GLiREL
from glirel.modules.run_evaluation import sample_train_data
from glirel.model import load_config_as_namespace
from glirel.modules.evaluator import greedy_search, RelEvaluator
from datetime import datetime
import json
import logging
import random
import shutil
import wandb
from functools import partial
from sklearn.model_selection import train_test_split
import time
import gc
import asyncio
import instructor
import openai
from enum import Enum
from pydantic import BaseModel, field_validator, Field
from asyncio import run as aiorun
from typing import List, Dict, Literal
import textwrap


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

'''

python eval_with_gpt.py --model gpt-4o \
    --eval-data data/few_rel_all.jsonl \
    --num-unseen-rel-types 5 \
    --seed 42

python eval_with_gpt.py \
    --predictions-file logs/gpt-wiki_zsl/wiki_zsl-2024-10-07__08-52-44/eval-predictions-gpt-4o-mini.jsonl \
        --eval-data data/wiki_zsl_all.jsonl
    
'''

client = instructor.from_openai(openai.AsyncOpenAI())


def create_parser():
    parser = argparse.ArgumentParser(description="Zero-shot Relation Extraction")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--eval-data", type=str, default=None, help="Path to evaluation data")
    parser.add_argument("--dataset-name", type=str, default=None, help="Name under which to save the logs")
    parser.add_argument("--num-unseen-rel-types", type=int, default=15, help="Number of unseen relation types for zero-shot learning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting data")
    parser.add_argument("--skip-splitting", action='store_true', help="Skip splitting data eval set")
    parser.add_argument("--predictions-file", type=str, default=None, help="output predictions path if it already exists")
    return parser


def get_unique_relations(data):
    unique_rel_types = []
    for item in data:
        for r in item['relations']:
            unique_rel_types.append(r["relation_text"])
    unique_rel_types = list(set(unique_rel_types))
    return unique_rel_types


def split_data_by_relation_type(data, num_unseen_rel_types, seed=None):
    """
    Attempts to split a dataset into training and testing sets based on relation types,
    aiming to have a specified number of unique relation types exclusively in the test set
    to simulate a zero-shot learning scenario. The function shuffles and splits the relation
    types, allocating the first chunk as unseen relation types for testing and the rest for training.
    
    It iteratively adjusts the number of unseen relation types if the initial split does not achieve
    the desired number of unique test relation types, retrying with an incremented number until it succeeds
    or the number reaches twice the original request, resetting as needed.

    Notes:
        - This function relies heavily on the assumption that sufficient relation diversity exists
          to meet the zero-shot criteria. If not, the test set may not end up with the intended
          number of unique unseen relation types.
        - The function can potentially skip a significant number of items that contain both train and
          test relation types, leading to data wastage.
        - The iterative process to adjust unseen relation types may lead to computational inefficiency,
          especially for large datasets with diverse relations.
    """

    unique_relations = get_unique_relations(data)
    correct_num_unseen_relations_achieved = False
    original_num_unseen_rel_types = num_unseen_rel_types

    logger.info(f"Running dataset splitting...")
    start = time.time()
    count = 0
    if seed is None:
        seed = random.randint(0, 1000)
    while correct_num_unseen_relations_achieved is False:
        random.seed(seed)
        random.shuffle(unique_relations)
        test_relation_types = set(unique_relations[ : num_unseen_rel_types ])
        train_relation_types = set(unique_relations[ num_unseen_rel_types : ])
        
        train_data = []
        test_data = []
        skipped_items = []
        
        # Splitting data based on relation types
        for item in data:
            relation_types = {r["relation_text"] for r in item['relations']}
            if relation_types.issubset(test_relation_types):
                test_data.append(item)
            elif relation_types.issubset(train_relation_types):
                train_data.append(item)
            else:
                # Entries that contain both train and test relation types are currently skipped
                skipped_items.append(item)
        
        # if we have the right number of eval relations, break
        if len(get_unique_relations(test_data)) == original_num_unseen_rel_types:
            correct_num_unseen_relations_achieved = True
        else:
            # bump the number of unseen relations by 1 to cast a wider net
            # if the bump gets too big, reset it
            num_unseen_rel_types = num_unseen_rel_types + 1 if (num_unseen_rel_types <  original_num_unseen_rel_types*2) else num_unseen_rel_types
            seed = random.randint(0, 1000)

        count += 1
        if count % 50 == 0:
            logger.info(f"Attempt {count} | Seed {seed}")

    if len(skipped_items) > 0:
        logger.info(f"Skipped items: {len(skipped_items)} because they have __BOTH__ train and test relation types")
    
    logger.info(f"Split on seed {seed}")
    logger.info(f"Splitting took {time.time() - start} seconds")
    return train_data, test_data
    
def dirty_split_data_by_relation_type(data, num_unseen_rel_types, max_test_size):
    '''
    This function does not care if the interesection of train and test relation types is empty.
    Used for custom datasets to avoid having a large number of eval classes (causes OOM), 
    and I do not mind if the eval set has some train classes.
    '''
    logger.info("Dirty splitting data...")

    unique_relations = get_unique_relations(data)
    correct_num_unseen_relations_achieved = False
    original_num_unseen_rel_types = num_unseen_rel_types


    while not correct_num_unseen_relations_achieved:
        seed = random.randint(0, 1000)
        random.seed(seed)
        random.shuffle(unique_relations)
        test_relation_types = set(unique_relations[ : num_unseen_rel_types ])
        
        train_data = []
        test_data = []

        # Splitting data based on relation types
        for item in data:
            relation_types = {r["relation_text"] for r in item['relations']}
            if len(test_data) < max_test_size and any([rel in test_relation_types for rel in relation_types]):
                test_data.append(item)
            else:
                train_data.append(item)

        # if we have the right number of eval relations, break
        if len(get_unique_relations(test_data)) == original_num_unseen_rel_types or len(test_data) >= max_test_size: 
            correct_num_unseen_relations_achieved = True
        else:
            # bump the number of unseen relations by 1 to cast a wider net
            # if the bump gets too big, reset it
            num_unseen_rel_types = num_unseen_rel_types + 1 if (num_unseen_rel_types <  original_num_unseen_rel_types*2) else num_unseen_rel_types


    return train_data, test_data


def eval_with_llm(model, log_dir, eval_rel_types, eval_data, predictions_file=None):

    async def async_eval_with_llm(eval_data, eval_rel_types, model, output_predictions):

        logger.info(f"🚀 Evaluating with LLM... 🚀")
        logger.info("Number of examples: ", len(eval_data))

        # eval_data = eval_data[:30]

        UNIQUE_LABELS = set(eval_rel_types)
        UNIQUE_LABELS.add("NO_RELATION")
        UNIQUE_LABELS = list(UNIQUE_LABELS)
        logger.info(f"UNIQUE LABELS: {UNIQUE_LABELS}")


        # use of Enum idea : https://stackoverflow.com/a/74335189
        class AnnotatedRelation(BaseModel):
            '''
            Classify the relationship between the HEAD and TAIL entities in the text,
            marked with [HEAD] and [TAIL] tokens respectively.
            Only use the relation labels provided to classify the relationship.
            If no relation exists, return 'NO_RELATION'
            '''
            head_entity: str
            tail_entity: str
            labels: List[str] = Enum("Labels", {l: l for l in UNIQUE_LABELS}, type=str)
            relation: List[labels] = Field(..., description="The relation between the head and tail entities. Before giving your answer, think carefully about this in the context of the given text.")

            @field_validator("relation")
            @classmethod
            def validate_relation(cls, relation: List[str]):
                if len(relation) > 1:
                    relation = [relation[0]]
                if 'NO_RELATION' in relation and len(relation) > 1:
                    raise ValueError("If 'NO_RELATION' is present, it should be the only label")
                elif any([label not in UNIQUE_LABELS for label in relation]):
                    raise ValueError(f"Invalid label found in relation: {relation}")
                return relation


        async def extract(message: Dict, model: str, output_predictions_path: str, text: str, relation: Dict):
            annotation = await client.chat.completions.create(
                model=model,
                messages=[
                    message,
                ],
                response_model=AnnotatedRelation,
                max_retries=3,
            )

            with open(output_predictions_path, "a+") as f:
                out = [text, relation, annotation.model_dump()['relation']]
                f.write(json.dumps(out) + "\n")
            
            return annotation

        async def batch_extract(messages: List[dict], model: str, output_predictions_path: str, texts: List[str], relations: List[Dict]) -> List["AnnotatedRelation"]:
            tasks = [
                extract(message, model, output_predictions_path, text, relation) for text, message, relation in zip(texts, messages, relations)
            ]
            results = await asyncio.gather(*tasks)
            return results


        prompt = textwrap.dedent(
        """
        Classify the relationship(s) between the HEAD and TAIL entities in the following text.
        Only use the relation labels provided to classify the relationship.
        If no relation exists, return ['NO_RELATION'].

        Text: {text}

        HEAD: {head}

        TAIL: {tail}

        Relation labels: {labels}
        """
        )
        messages = []
        text2messasge = {}
        for item in eval_data:
            for rel in item['relations']:
                head = rel['head']['mention']
                tail = rel['tail']['mention']
                text = " ".join(item['tokenized_text'])
                message = {
                    "role": "user", 
                    "content": prompt.format(text=text, head=head, tail=tail, labels=UNIQUE_LABELS)
                }
                messages.append(
                    {
                        "message": message,
                        "text": text,
                        'relation': rel,
                    }
                )
                if text in text2messasge:
                    text2messasge[text].append([messages[-1], rel['relation_text']])
                else:
                    text2messasge[text] = [[messages[-1], rel['relation_text']]]

        logger.info(f"Number of messages for {len(eval_data)} examples: {len(messages)}")
        batch_size = 100
        logger.info(f"🚀 Annotating {len(messages)} examples... 🚀")
        for batch in tqdm(range(0, len(messages), batch_size)):
            batch = messages[batch : batch+batch_size]
            batch_messages = [b['message'] for b in batch]
            batch_texts = [b['text'] for b in batch]
            batch_relations = [b['relation'] for b in batch]
            assert len(batch_messages) == len(batch_texts) == len(batch_relations)
            batch_messages = await batch_extract(
                batch_messages, 
                model=model, 
                output_predictions_path=output_predictions,
                texts=batch_texts,
                relations=batch_relations
            )

        logger.info(f"Annotation written to {output_predictions}")
        return
    
    if predictions_file is None:
        output_predictions = f"{log_dir}/eval-predictions-{model}.jsonl"
        asyncio.run(async_eval_with_llm(eval_data, eval_rel_types, model, output_predictions))
    else:
        logger.info(f"Predictions file already exists. Skipping annotation")
        output_predictions = predictions_file

    with open(output_predictions, "r") as f:
        predictions = [json.loads(line) for line in f]

    text2rel = {}
    for d in predictions:
        if d[0] not in text2rel:
            text2rel[d[0]] = [d[1]]
        else:
            text2rel[d[0]].append(d[1])
    
    gt_two_rels = [t for t,v in text2rel.items() if len(v) > 1]
    print(f"Texts with more than 2 relations: {gt_two_rels}")

    text2preds = {}
    text2trues = {}
    for idx, p in enumerate(predictions):

        if p[0] not in text2trues:
            text2trues[p[0]] = []
        text2trues[p[0]].append(p[1])

        if p[0] not in text2preds:
            text2preds[p[0]] = []
        if p[-1] in [['NO_RELATION'], []]:
            continue
        rel = {
            'head_text': p[1]['head']['mention'],
            'tail_text': p[1]['tail']['mention'],
            'head' : {'position': p[1]['head']['position']},
            'tail' : {'position': p[1]['tail']['position']},
            'relation_text': p[-1][0],
            'score': 1.0,
        }


        text2preds[p[0]].append(rel)


    preds = list(text2preds.values())
    all_trues = list(text2trues.values())
    for p, true, in zip(preds, all_trues):
            print(p, "-->", true)
    evaluator = RelEvaluator(all_trues, preds)
    out, metric_dict = evaluator.evaluate()
    logger.info(f"Metrics: {metric_dict}")


def main(args):

    if args.eval_data is not None:
        eval_data = args.eval_data

    if args.dataset_name is None:
        if 'wiki_zsl' in args.eval_data:
            dataset_name = 'wiki_zsl'
        elif 'redocred' in args.eval_data:
            dataset_name = 'redocred'
        elif 'fewrel' in args.eval_data or 'few_rel' in args.eval_data:
            dataset_name = 'fewrel'
        else:
            raise ValueError(f"Could not find dataset name from arg: {args.eval_data}. Please provide dataset name in --dataset-name")
    else:
        dataset_name = args.dataset_name

    # set up logging
    current_time = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    log_dir = f'logs/gpt-{dataset_name}/{dataset_name}-{current_time}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = "eval.log"
    log_file_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Relation extraction evlauation started")
    logger.info(f"Evaluating on file {eval_data}")


    if isinstance(eval_data, str):
        eval_data_list = [eval_data]

    eval_data = []
    for eval_subset in eval_data_list:
        if eval_subset.endswith('.jsonl'):
            with open(eval_subset, 'r') as f:
                eval_subset = [json.loads(line) for line in f]
        elif eval_subset.endswith('.json'):
            with open(eval_subset, 'r') as f:
                eval_subset = json.load(f)
        else:
            raise ValueError(f"Invalid data format: {eval_data}. Must be .jsonl or .json")
        eval_data.extend(eval_subset)

    if args.skip_splitting is False:
        _, eval_data = split_data_by_relation_type(eval_data, args.num_unseen_rel_types, seed=args.seed)


    eval_rel_types = get_unique_relations(eval_data)
    if eval_data is not None:
        logger.info(f"Num Eval relation types: {len(eval_rel_types)}")
        logger.info(f"Number of eval samples: {len(eval_data)}")

    if len(eval_rel_types) != args.num_unseen_rel_types:
        logger.info(f"Num eval types not set. Will take them on a batch by batch bases")
        eval_rel_types = None

    eval_with_llm(
        args.model, 
        log_dir, 
        eval_rel_types=eval_rel_types, 
        eval_data=eval_data,
        predictions_file=args.predictions_file
    )


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    main(args)
