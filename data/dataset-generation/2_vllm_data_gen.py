
from datasets import load_dataset
from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
from datasets import Dataset, concatenate_datasets
import pandas as pd
import torch
import huggingface_hub
import textwrap
from datasets import Dataset, concatenate_datasets

import time, tqdm
import sys

# memory utils
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel
import os
import ray


'''
    Usage:
        python 2-vllm_data_gen.py 2048 "jackboyla/fineweb_spacy" "jackboyla/ZeroRel" "CC-MAIN-2023-06" "mistralai/Mistral-7B-Instruct-v0.2" <HF_TOKEN>
    args:
        int(INFERENCE_BATCH_SIZE), 
        INFER_REPO_ID, 
        REPO_ID, 
        PARTITION_NAME, 
        MODEL_ID, 
        HF_TOKEN, 
'''


### memory utils
# memory clearing sorcery from https://github.com/vllm-project/vllm/issues/1908

def clear_memory() -> None:
    """Clears the memory of unused items."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


INTERESTING_TYPES = [
    "PERSON",   # People, including fictional
    "NORP",     # Nationalities or religious or political groups
    "FAC",      # Facilities like buildings, airports, highways, bridges
    "ORG",      # Organizations, including companies, agencies, institutions
    "GPE",      # Countries, cities, states
    "LOC",      # Non-GPE locations, mountain ranges, bodies of water
    "PRODUCT",  # Objects, vehicles, foods, etc. (not services)
    "EVENT",    # Named hurricanes, battles, wars, sports events, etc.
    "WORK_OF_ART", # Titles of books, songs, etc.
    "LAW",      # Named documents made into laws.
    "LANGUAGE", # Any named language
    "DATE",     # Absolute or relative dates or periods
    "TIME",     # Times smaller than a day
    "MONEY",    # Monetary values, including unit
    "ORDINAL",  # "first", "second", etc.
    "CARDINAL", # Numerals that do not fall under another type
]

''' 
I tried batching entity pairs together for each inference,
but it's currently just as slow as doing single pair inferences
'''

def generate_entity_pairs_data(ds, current_idx=0, NUM_PAIRS_TOGETHER=1):
    ''' 
        get all entity pairs (all possible relations)
        Checks entity type
        If the head/tail is in `INTERESTING_TYPES`, keep the candidate
        This is done to reduce the number of redundant pairs
    '''

    DATA = []
    ent_labels = set()

    start = time.time()

    iter_ds = ds.iter(batch_size=1)

    for idx, doc in enumerate(iter_ds):

        ner = doc["ents"][0]
        temp_ents = []

        # Store common document data once per document
        current_idx += 1
        current_text_data = {
                'id': current_idx,
                'text': doc["text"],
                'tokenized_text': doc["tokens"],
                'ner': doc['ents']
            }

        # Collect the "interesting" entity pairs
        for head_ent in ner:
            for tail_ent in ner:
                if head_ent != tail_ent and (head_ent[2] in INTERESTING_TYPES or tail_ent[2] in INTERESTING_TYPES):
                    pair = {'head': head_ent, 'tail': tail_ent}
                    ent_labels.add(head_ent[2])
                    ent_labels.add(tail_ent[2])
                    temp_ents.append(pair)

        # Group data in batches of num_pairs_together
        while len(temp_ents) >= NUM_PAIRS_TOGETHER:
            DATA.append({
                **current_text_data,
                'ents': temp_ents[ : NUM_PAIRS_TOGETHER ]
            })
            temp_ents = temp_ents[NUM_PAIRS_TOGETHER : ]


    print(f"Entity pair generation duration --> {time.time() - start}")
        
    return DATA


# format prompt
# for Mistral 7B Instruct 
# --> https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct

def format_prompts(DATA, MODEL_ID):
    '''Format the prompt for the model to generate the relation label'''
    template = textwrap.dedent(
        """You are a fantastic relation extraction model who only outputs valid JSON.
        Extract the relation between the given entities using the context in the below text. If no relation exists, use the label NO_RELATION.
        ONLY RETURN THE RELATION LABEL. Do not add additional text.
        Pay VERY close attention to which entity is the head and tail; this dictates the direction of the relationship.

        Text: 
        {text}

        Entities: 
        {ents}

        Relation: """
    )


    for d in DATA:
    #     formatted_ents = {f"ent_pair_{i}": ent for i, ent in enumerate(d["ents"]) }
        d['instruction'] = template.format(
            text=d['text'], 
            ents=d["ents"]
        )
        d["model_name"] = MODEL_ID

    return DATA



# when the pairs have been annotated INDIVIDUALLY by the LLM,
# we want to aggregate them back into their respective documents

# aggregate dataset
def aggregate_dataset(dataset):
    
    '''
    keys([
    'id', 'text', 'ents', 'model_name' 'tokenized_text', 'instruction', 'generation'
    ])
    '''
    
    df = dataset.to_pandas()

    # Define a custom aggregation function for the ner column
    def first_ner(ner_lists):
        # Assuming we want the first nested sequence directly
        if len(ner_lists) > 0:
            return ner_lists.iloc[0][0]
        return None

    # Aggregate the data
    aggregated_df = df.groupby('id').agg({
        'text': 'first',                # Keep the first text
        "tokenized_text": "first",  
        "model_name": "first",  
        "instruction": "first", 
        "ner": first_ner,
        'ents': lambda x: list(x),      # Combine all ents into one list
        'generation': lambda x: list(x) # Combine all outputs into one list
    }).reset_index()

    # Convert the aggregated DataFrame back to a HuggingFace dataset if needed
    aggregated_dataset = Dataset.from_pandas(aggregated_df)
    return aggregated_dataset




# from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_distributed.py

# Create a class to do batch inference.
class LLMPredictor:

    def __init__(self, MODEL_ID, REPO_ID, pydantic_model=None, existing_ds=None):
        
        NUM_GPUS = torch.cuda.device_count()
        print(f"NUM GPUs: {NUM_GPUS}")
        assert NUM_GPUS > 0, "Must have at least one GPU to use 😤"

        self.REPO_ID = REPO_ID

        # 📖 Model Configuration
        self.llm = LLM(
            MODEL_ID,
            dtype = 'half', 
            tensor_parallel_size = NUM_GPUS, 
            max_model_len = 1_000,
            gpu_memory_utilization = 0.9,
            enable_prefix_caching=True,
        )
#         logits_processor = JSONLogitsProcessor(schema=pydantic_model, llm=self.llm.llm_engine)
        
        
        self.sampling_params = SamplingParams(
            temperature = 0.1, 
            max_tokens = 20,
#             logits_processors=[logits_processor]
        )
        
        self.existing_ds = existing_ds
        self.hf_ds = None
        
        self.push_counter = 0   
        self.push_interval = 15  # num batches before pushing to HF

        
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["instruction"], self.sampling_params)
        generated_text = []
        for output in outputs:
            generated_text.append(' '.join([o.text for o in output.outputs]))
        batch['generation'] = generated_text
        
        batch_ds = Dataset.from_pandas(pd.DataFrame(data=batch))
        
        if self.hf_ds is None:
            self.hf_ds = batch_ds
        else:
            self.hf_ds = concatenate_datasets([self.hf_ds, batch_ds])
        
        self.push_counter += 1
        if self.push_counter % self.push_interval == 0:
            
            # aggregate by `id` column
            aggregated_ds = aggregate_dataset(self.hf_ds)
            if self.existing_ds is not None:
                aggregated_ds = concatenate_datasets([self.existing_ds, aggregated_ds])
            aggregated_ds.push_to_hub(self.REPO_ID)
        
        return batch

    

def main(
	INFERENCE_BATCH_SIZE: int, 
	INFER_REPO_ID: str, 
	REPO_ID: str, 
	PARTITION_NAME: str, 
	MODEL_ID: str, 
	HF_TOKEN: str, 
):
    local_params = {k:v for k,v in locals().items() if 'token' not in k.lower()}
    print(f"All params: {local_params}")

    # log in to 🤗 hub, needed for gated model repos
    huggingface_hub.login(token=HF_TOKEN)
    
    try:
        existing_dataset = load_dataset(REPO_ID, split="train")
    except:
        print(f"Dataset {REPO_ID} does not exist yet!")
        existing_dataset = None
    

    # load existing RE dataset (if it exists)
    # --> for appending to existing data
    if existing_dataset is not None:
        # get the max doc ID so we know what ID to assign new docs
        
        EXISTING_MAX_ID = max(existing_dataset['id'])
        print(f"EXISTING_MAX_ID: {EXISTING_MAX_ID}")
    else:
        EXISTING_MAX_ID = 0


    # load dataset to run inference on
    fw = load_dataset(INFER_REPO_ID, split=PARTITION_NAME)

    # remove already annotated items
    ds = fw.select(range(EXISTING_MAX_ID, min(EXISTING_MAX_ID, len(fw))))

    DATA = generate_entity_pairs_data(ds, current_idx = EXISTING_MAX_ID)
    
    DATA = format_prompts(DATA, MODEL_ID=MODEL_ID)
    
    print(f"number of ent-ent pairs --> {len(DATA)}!")
    
    instruction_data = {k: [] for k in DATA[0].keys()}
    for d in DATA:
        for k, v in d.items():
            instruction_data[k].append(v)


    llm_pipe = LLMPredictor(MODEL_ID=MODEL_ID, REPO_ID=REPO_ID, existing_ds = existing_dataset)
            
    # run inference
    for i in range(0, len(instruction_data['id']), INFERENCE_BATCH_SIZE):

        batch_num = int(i // INFERENCE_BATCH_SIZE)
        total_num_batches = int(len(instruction_data['id']) // INFERENCE_BATCH_SIZE)
        print(f"Beginning batch {batch_num}/{total_num_batches} at example # {i}!")

        batch = {}
        for k in instruction_data.keys():
            batch[k] = [v for v in instruction_data[k][ i : i+INFERENCE_BATCH_SIZE ]]

        batch_out = llm_pipe(batch)


        print(f"batch_out['generation'][0] --> {batch_out['generation'][0]}")


    # final push
    aggregated_ds = aggregate_dataset(llm_pipe.hf_ds)
    if llm_pipe.existing_ds:
        aggregated_ds = concatenate_datasets([llm_pipe.existing_ds, aggregated_ds])
    aggregated_ds.push_to_hub(REPO_ID)
    
    
	# CLEAR MEMORY
    #llm_pipe.llm is a vllm.LLM object

    #avoid huggingface/tokenizers process deadlock
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    destroy_model_parallel()
    gc.collect()
    #del a vllm.executor.ray_gpu_executor.RayGPUExecutor object
    # llm_pipe.llm.llm_engine.model_executor.shutdown()
    del llm_pipe.llm.llm_engine.model_executor.driver_worker
    del llm_pipe.llm.llm_engine.model_executor
    del llm_pipe.llm
    del llm_pipe
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
	
    clear_memory()
    gc.collect()



if __name__ == "__main__":
   
    INFERENCE_BATCH_SIZE, REPO_ID, INFER_REPO_ID, PARTITION_NAME, MODEL_ID, HF_TOKEN, APPEND_TO_REPO_ID = sys.argv[1:]
    main(
        int(INFERENCE_BATCH_SIZE), 
        INFER_REPO_ID, 
        REPO_ID, 
        PARTITION_NAME, 
        MODEL_ID, 
        HF_TOKEN, 
    )
