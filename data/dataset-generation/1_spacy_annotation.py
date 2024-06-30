import huggingface_hub
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import spacy
import torch
import time
from tqdm import tqdm
import gc
import sys


'''
    Usage:
        python 1-spacy_annotation.py "CC-MAIN-2023-06" "jackboyla/spacy_fineweb" 1000 256 <HF_TOKEN>
    args:
        - FineWeb partition_name, 
        - REPO_ID to push to, 
        - int(DATASET_SIZE), 
        - int(MAX_SEQ_LEN),
        - HF_TOKEN, 
'''
def main(partition_name, REPO_ID, DATASET_SIZE, MAX_SEQ_LEN, HF_TOKEN):

    huggingface_hub.login(token=HF_TOKEN)

    # spacy.require_gpu()
    nlp = spacy.load('en_core_web_trf')

    # Load FineWeb dataset
    # ALL FineWeb partitions --> https://huggingface.co/datasets/HuggingFaceFW/fineweb#breakdown-by-dumpcrawl

    fw = load_dataset("HuggingFaceFW/fineweb", name=partition_name, split="train", streaming=True)

    PARTITION_NAME = "_".join(partition_name.split('-'))


    # 🤗 The Huggingface hub dataset repo to push to
    existing_ds = load_dataset(REPO_ID)
    existing_ds


    def convert_to_hf_ds(docs):
        data = {
        "text": [doc.text for doc in docs],
        "tokens": [[token.text for token in doc] for doc in docs],
        "ents": [[[str(e.start), str(e.end), e.label_, e.text] for e in doc.ents] for doc in docs]
        }

        # Create a Hugging Face dataset
        dataset = Dataset.from_dict(data)

        return dataset


    # if there's already an existing partition for this split,
    # find out how many examples have already been processed

    if PARTITION_NAME in existing_ds:
        existing_docs = existing_ds[PARTITION_NAME]
        NUM_EXISTING_DOCS = len(existing_docs)
    else:
        existing_docs = None
        NUM_EXISTING_DOCS = 0

    print(f"{NUM_EXISTING_DOCS} docs already processed for this partition: {PARTITION_NAME}")




    # collect texts of < MAX_SEQ_LEN

    texts = []
    dataset = iter(fw)
    end_reached = False

    i = 0

    with tqdm(total=DATASET_SIZE) as pbar:
        
        while len(texts) < DATASET_SIZE and end_reached is False:
            
            try:
                
                example = next(dataset)
                if i < NUM_EXISTING_DOCS or example['token_count'] > MAX_SEQ_LEN:
                    pass
                else:
                    texts.append(example['text'])
                    pbar.update(1)
                    
            except:
                end_reached = True

            i += 1

    print(len(texts))


    # run spacy NER

    UPLOAD_EVERY = 1000
    BATCH_SIZE = 100

    start = time.time()

    docs = []

    i = 1

    with tqdm(total=DATASET_SIZE) as pbar:
        
        for j in range(0, DATASET_SIZE, BATCH_SIZE):
            
            texts_batch = texts[j : min(j + BATCH_SIZE, len(texts))]

            for doc in nlp.pipe(texts_batch, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):

                ents = set([e.label_ for e in doc.ents])
                if 'PERSON' in ents and len(ents) > 2:  # want texts with people in them
                    docs.append(doc)
                    pbar.update(1)
                    i += 1


                if i % UPLOAD_EVERY == 0:
                    ds = convert_to_hf_ds(docs)

                    if existing_docs:
                        ds = concatenate_datasets([existing_docs, ds])

                    existing_ds = load_dataset(REPO_ID, download_mode='force_redownload')
                    existing_ds[PARTITION_NAME] = ds
                    check_for_duplicates(existing_ds)
                    existing_ds.push_to_hub(REPO_ID)
                    
                    docs = []   # Clear docs after uploading to prevent duplicates
                    i += 1


    print(f"Data Collection Duration: {time.time() - start}")


def check_for_duplicates(dataset: Dataset) -> None:

    for k in dataset.keys():
        total_texts = {}
        for row in dataset[k]:
            if row['text'] in total_texts:
                print(f"DUPLICATE: {row['text']}")
            else:
                total_texts[row['text']] = 1
    return



if __name__ == "__main__":
   
    partition_name, REPO_ID, DATASET_SIZE, MAX_SEQ_LEN, HF_TOKEN = sys.argv[1:]
    main(
        partition_name, 
        REPO_ID, 
        int(DATASET_SIZE), 
        int(MAX_SEQ_LEN),
        HF_TOKEN, 
    )