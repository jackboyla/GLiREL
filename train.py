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

python train.py --config config_wiki_zsl.yaml


python train.py --config config_few_rel.yaml


'''

# If doing hyperparameter sweeping, define sweep config here

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "eval_f1"},
    "parameters": {
        "num_train_rel_types": {"values": [15, 20, 25, 30, 35, 40]},
        "num_unseen_rel_types": {"values": [15]},
        "random_drop": {"values": [True, False]},
        "lr_others": {"max": 1e-3, "min": 5e-5},
        "dropout": {"max": 0.55, "min": 0.3},
        "model_name": {"values": ["microsoft/deberta-v3-large", "microsoft/deberta-v3-small"]},
    },
}


def create_parser():
    parser = argparse.ArgumentParser(description="Zero-shot Relation Extraction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the log directory')
    parser.add_argument("--wandb_log", action="store_true", help="Activate wandb logging")
    parser.add_argument("--wandb_sweep", action="store_true", help="Activate wandb hyperparameter sweep")
    parser.add_argument("--skip_splitting", action="store_true", help="Skip dataset splitting into train and eval sets")
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

    
def dirty_split_data_by_relation_type(data, num_unseen_rel_types, max_test_size=3000):
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
        if len(get_unique_relations(test_data)) == original_num_unseen_rel_types: 
            correct_num_unseen_relations_achieved = True
        else:
            # bump the number of unseen relations by 1 to cast a wider net
            # if the bump gets too big, reset it
            num_unseen_rel_types = num_unseen_rel_types + 1 if (num_unseen_rel_types <  original_num_unseen_rel_types*2) else num_unseen_rel_types


    return train_data, test_data


# train function
def train(model, optimizer, train_data, config, eval_data=None, num_steps=1000, eval_every=100, top_k=1, log_dir=None,
          wandb_log=False, wandb_sweep=False, warmup_ratio=0.1, train_batch_size=8, device='cuda', use_amp=True):
    
    train_rel_types = get_unique_relations(train_data)
    eval_rel_types = get_unique_relations(eval_data) if eval_data is not None else None
    max_saves = 2  # Maximum number of saved models

    if wandb_log:
        # Start a W&B Run with wandb.init
        wandb.login()
        run = wandb.init()
    else:
        run = None
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()

    # initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=train_batch_size, shuffle=False, train_relation_types=train_rel_types)

    pbar = tqdm(range(num_steps))

    if warmup_ratio < 1:
        num_warmup_steps = int(num_steps * warmup_ratio)
    else:
        num_warmup_steps = int(warmup_ratio)

    if config.scheduler == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif config.scheduler == "cosine_with_hard_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps,
            num_cycles=3
        )
    else:
        raise ValueError(f"Invalid scheduler: {config.scheduler}")

    iter_train_loader = iter(train_loader)

    saved_models = []
    best_f1 = 0

    accumulated_steps = 0 

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}


        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            try:
                loss = model(x)  # Forward pass
            except Exception as e:
                logger.error(f"Error in step {step}: {e}")
                logger.error(f"Num tokens: {[len(x['tokens'][i]) for i in range(len(x['tokens']))]}")
                logger.error(f"Num relations: {[x['rel_label'][i].shape[0] for i in range(len(x['rel_label']))]}")
                logger.error(f"Num spans: {[x['span_idx'][i].shape[0] for i in range(len(x['span_idx']))]}")
                logger.error(f"Num candidate classes: {[len(x['classes_to_id'][i]) for i in range(len(x['classes_to_id']))]}")
                continue
        

        # check if loss is nan
        if torch.isnan(loss):
            logger.warn(f"Loss is NaN at step {step}")
            continue

        if config.gradient_accumulation is not None:
            loss = loss / config.gradient_accumulation  # Normalize the loss to account for the accumulation

        try:
            scaler.scale(loss).backward()  # Compute gradients
        except Exception as e:
            logger.error(f"Backprop Loss Error in step {step}: {e}")
            logger.error(f"Num tokens: {[len(x['tokens'][i]) for i in range(len(x['tokens']))]}")
            logger.error(f"Num relations: {[x['rel_label'][i].shape[0] for i in range(len(x['rel_label']))]}")
            logger.error(f"Num spans: {[x['span_idx'][i].shape[0] for i in range(len(x['span_idx']))]}")
            logger.error(f"Num candidate classes: {[len(x['classes_to_id'][i]) for i in range(len(x['classes_to_id']))]}")
            continue

        logger.info(f"Step {step} | loss: {loss.item()} | x['rel_label']: {x['rel_label'].shape} | x['span_idx']: {x['span_idx'].shape} | x['tokens']: {[len(x['tokens'][i]) for i in range(len(x['tokens']))]} | num candidate_classes: {len(x['classes_to_id'][0].keys())}")

        accumulated_steps += 1
        if config.gradient_accumulation is None or (accumulated_steps % config.gradient_accumulation == 0):
            # optimizer.step()        # Update parameters
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()        # Update learning rate schedule
            optimizer.zero_grad(set_to_none=True)   # Clear gradients after update (set_to_none=True here can modestly improve performance)
            accumulated_steps = 0   # Reset accumulation counter


        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"

        if run is not None:
            run.log({"loss": loss.item()})

        elif wandb_sweep:
            wandb.log(
                    {
                    "epoch": step // len(train_loader),
                    "train_loss": loss.item(),
                }
            )

        if (step + 1) % eval_every == 0:

            model.eval()

            current_path = os.path.join(log_dir, f'model_{step + 1}')
            model.save_pretrained(current_path)
            logger.info(f"Model saved at {current_path}")

            if eval_data is None:
                saved_models.append(current_path)
                if len(saved_models) > max_saves:
                    oldest_model = saved_models.pop(0)
                    shutil.rmtree(oldest_model)
            
            elif eval_data is not None:
                with torch.no_grad():

                    logger.info('Evaluating...')
                    logger.info(f'Taking top k = {top_k} predictions for each relation...')

                    results, f1 = model.evaluate(
                        eval_data, 
                        flat_ner=True, 
                        threshold=config.eval_threshold, 
                        batch_size=config.eval_batch_size,
                        entity_types=eval_rel_types,
                        top_k=top_k
                    )

                    if wandb_sweep:
                        wandb.log(
                                {
                                "epoch": step // len(train_loader),
                                "eval_f1": f1,
                            }
                        )
                    elif run is not None:
                        run.log({"eval_f1": f1})

                    logger.info(f"Step={step}\n{results}")
                    

                    saved_models.append((current_path, f1))
                    if len(saved_models) > max_saves:
                        saved_models.sort(key=lambda x: x[1], reverse=True)  # Sort models by F1 score
                        lowest_f1_model = saved_models.pop()  # Remove the model with the lowest F1 score
                        if lowest_f1_model[1] < best_f1:
                            shutil.rmtree(lowest_f1_model[0])  # Delete the model file if its score is the lowest
                        
                        best_f1 = max(best_f1, f1)  # Update the best score
            

            model.train()
                
            torch.cuda.empty_cache()  # Clear cache after evaluation to prepare for training
            gc.collect()

        pbar.set_description(description)


def main(args):

    # load config
    config = load_config_as_namespace(args.config)

    config.log_dir = args.log_dir

    # set up logging
    if config.log_dir is None:
        current_time = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        config.log_dir = f'logs/{config.dataset_name}/{config.dataset_name}-{current_time}'
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    log_file = "train.log"
    log_file_path = os.path.join(config.log_dir, log_file)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Relation extraction training started")


    if args.wandb_sweep:
        run = wandb.init()
        # overwrite config values with sweep values 
        config.num_train_rel_types = wandb.config.num_train_rel_types
        config.num_unseen_rel_types = wandb.config.num_unseen_rel_types
        config.lr_others = wandb.config.lr_others


    # Prep data

    if config.train_data.endswith('.jsonl'):
        with open(config.train_data, 'r') as f:
            data = [json.loads(line) for line in f]
            # data = []
            # for i in range(50_000):
            #     data.append(json.loads(next(f)))
    elif config.train_data.endswith('.json'):
        with open(config.train_data, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Invalid data format: {config.train_data}")



    if hasattr(config, 'eval_data'):

        if config.eval_data.endswith('.jsonl'):
            with open(config.eval_data, 'r') as f:
                eval_data = [json.loads(line) for line in f]
        elif config.eval_data.endswith('.json'):
            with open(config.eval_data, 'r') as f:
                eval_data = json.load(f)
        else:
            raise ValueError(f"Invalid data format: {config.eval_data}. Must be .jsonl or .json")

    else:
        eval_data = None


    # train / eval split

    if eval_data is None:
        if args.skip_splitting:
            print("Skipping dataset splitting")
            data = sorted(data, key=lambda x: len(x['relations']))
            train_data = data
        elif config.num_unseen_rel_types is not None:
            # create eval set from train data
            if config.dataset_name == 'zero_rel':
                # train_data, eval_data = dirty_split_data_by_relation_type(data, config.num_unseen_rel_types)
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
            print("No unseen relation types specified. Randomly splitting data into train and eval sets.")
            data = sorted(data, key=lambda x: len(x['relations']))
            train_data, eval_data = train_test_split(data, test_size=5_000, random_state=42)
    else:
        # _, eval_data = split_data_by_relation_type(eval_data, config.num_unseen_rel_types)
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

    if config.prev_path != "none":
        model = GLiREL.from_pretrained(config.prev_path)
        model.config = config
    else:
        model = GLiREL(config)

    # Get number of parameters (trainable and total)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_params} / {num_params}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device != 'cpu' 
    model = model.to(device)

    lr_encoder = float(config.lr_encoder)
    lr_others = float(config.lr_others)

    optimizer = torch.optim.AdamW([
        # encoder
        {'params': model.token_rep_layer.parameters(), 'lr': lr_encoder},
        {'params': model.rnn.parameters(), 'lr': lr_others},
        # projection layers
        {'params': model.span_rep_layer.parameters(), 'lr': lr_others},
        {'params': model.prompt_rep_layer.parameters(), 'lr': lr_others},
    ])

    logger.info(f"Using config: \n{json.dumps(config.__dict__, indent=2)}\n\n")


    train(model, optimizer, data, config, eval_data=eval_data, num_steps=config.num_steps, eval_every=config.eval_every, top_k=config.top_k,
          log_dir=config.log_dir, wandb_log=args.wandb_log, wandb_sweep=args.wandb_sweep, warmup_ratio=config.warmup_ratio, train_batch_size=config.train_batch_size,
          device=device, use_amp=use_amp)


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    assert not (args.wandb_log is True and args.wandb_sweep is True), "Cannot use both wandb logging and wandb sweep at the same time."

    if args.wandb_sweep:
        # get day and time as string
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        sweep_name = f"sweep-{dt_string}"
        sweep_configuration["name"] = sweep_name


        # Initialize sweep by passing in config
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="GLiREL")

        # Start sweep job
        wandb.agent(sweep_id, function=partial(main, args), count=10)
    else:
        main(args)
