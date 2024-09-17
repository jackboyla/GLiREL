import argparse
import os

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

# from model_nested import NerFilteredSemiCRF
from glirel import GLiREL
from glirel.modules.run_evaluation import sample_train_data
from glirel.model import load_config_as_namespace
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


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

'''

python eval.py --ckpt-dir logs/docred/docred-2024-09-16__13-06-14/model_19500 \
    --eval-data data/redocred_test.jsonl

'''

def create_parser():
    parser = argparse.ArgumentParser(description="Zero-shot Relation Extraction")
    parser.add_argument("--ckpt-dir", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    return parser


def get_unique_relations(data):
    unique_rel_types = []
    for item in data:
        for r in item['relations']:
            unique_rel_types.append(r["relation_text"])
    unique_rel_types = list(set(unique_rel_types))
    return unique_rel_types



def split_data_by_relation_type(data, num_unseen_rel_types):
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
    while not correct_num_unseen_relations_achieved:
        seed = random.randint(0, 1000)
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
        # logger.info('Incorrect number of unseen relation types. Retrying...')

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


# train function
def eval(model, config, eval_rel_types, eval_data, 
         top_k=1,
         device='cuda', use_amp=True):

        model.eval()
    
        with torch.no_grad():

            logger.info('Evaluating...')
            logger.info(f'Taking top k = {top_k} predictions for each relation...')

            results, micro_f1, macro_f1 = model.evaluate(
                eval_data, 
                flat_ner=True, 
                threshold=config.eval_threshold, 
                batch_size=config.eval_batch_size,
                relation_types=eval_rel_types if config.fixed_relation_types else [],
                top_k=top_k
            )

            logger.info(f"Results = {results}")


        torch.cuda.empty_cache()  # Clear cache after evaluation to prepare for training
        gc.collect()


def main(args):

    # load config
    config_path = args.ckpt_dir + '/glirel_config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        config = argparse.Namespace(**config_dict)

    if args.eval_data is not None:
        config.eval_data = args.eval_data

    # set up logging
    current_time = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    config.log_dir = f'logs/{config.dataset_name}/{config.dataset_name}-{current_time}'
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    log_file = "eval.log"
    log_file_path = os.path.join(config.log_dir, log_file)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Relation extraction evlauation started")
    logger.info(f"Evaluating on file {config.eval_data}")



    # Prep data

    if isinstance(config.train_data, str):
        config.train_data = [config.train_data]

    train_data = []
    for train_subset in config.train_data:
        if train_subset.endswith('.jsonl'):
            with open(train_subset, 'r') as f:
                train_subset = [json.loads(line) for line in f]
                # train_subset = []
                # for i in range(1_000):
                #     train_subset.append(json.loads(next(f)))
        elif train_subset.endswith('.json'):
            with open(train_subset, 'r') as f:
                train_subset = json.load(f)
        else:
            raise ValueError(f"Invalid data format: {config.train_data}")
        train_data.extend(train_subset)
    data = train_data



    if hasattr(config, 'eval_data'):

        if isinstance(config.eval_data, str):
            config.eval_data = [config.eval_data]

        eval_data = []
        for eval_subset in config.eval_data:
            if eval_subset.endswith('.jsonl'):
                with open(eval_subset, 'r') as f:
                    eval_subset = [json.loads(line) for line in f]
            elif eval_subset.endswith('.json'):
                with open(eval_subset, 'r') as f:
                    eval_subset = json.load(f)
            else:
                raise ValueError(f"Invalid data format: {config.eval_data}. Must be .jsonl or .json")
            eval_data.extend(eval_subset)

    else:
        eval_data = None


    # train / eval split

    if eval_data is None:
        if args.skip_splitting:
            print("Skipping dataset splitting. Randomly splitting data into train and eval sets.")
            data = sorted(data, key=lambda x: len(x['relations']))
            
        elif config.num_unseen_rel_types is not None:

            if 'zero_rel' in config.dataset_name:
                file_name = 'data/wiki_zsl_all.jsonl'
                config.eval_data = file_name
                with open(file_name, 'r') as f:
                    logger.info(f"Generating eval split from {file_name}...")
                    eval_data = [json.loads(line) for line in f]
                _, eval_data = split_data_by_relation_type(eval_data, config.num_unseen_rel_types)
                data = sorted(data, key=lambda x: len(x['relations']))
                train_data = data
            else:
                train_data, eval_data = split_data_by_relation_type(data, config.num_unseen_rel_types)
        else:
            raise ValueError("No eval data provided and config.num_unseen_rel_types is None")
    else:
        eval_data = eval_data
        train_data = data

    train_rel_types = get_unique_relations(train_data)
    eval_rel_types = get_unique_relations(eval_data) if eval_data is not None else None
    logger.info(f"Num Train relation types: {len(train_rel_types)}")
    logger.info(f"Number of train samples: {len(train_data)}")
    if eval_data is not None:
        logger.info(f"Intersection: {set(train_rel_types) & set(eval_rel_types)}")
        logger.info(f"Num Eval relation types: {len(eval_rel_types)}")
        logger.info(f"Number of eval samples: {len(eval_data)}")


    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GLiREL.from_pretrained(args.ckpt_dir, map_location=device)
    model.config = config

    # Get number of parameters (trainable and total)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_params} / {num_params}")

    use_amp = device != 'cpu' 
    model = model.to(device)

    logger.info(f"Using config: \n{json.dumps(config.__dict__, indent=2)}\n\n")


    eval(model, config, eval_rel_types=eval_rel_types, eval_data=eval_data,
          top_k=config.top_k,
          device=device)


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    main(args)
