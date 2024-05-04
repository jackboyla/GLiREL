# GLiREL : Generalist and Lightweight model for Zero-Shot Relation Extraction

GLiREL is a Relation Extraction model capable of classifying unseen relations given the entities within a text. This builds upon the excelent work done by Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois on the [GLiNER](https://github.com/urchade/GLiNER) library which enables efficient zero-shot Named Entity Recognition.

* GLiNER paper: [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526)

* Train a Zero-shot model: <a href="https://colab.research.google.com/github/jackboyla/GLiREL/blob/main/train.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!-- <img src="demo.jpg" alt="Demo Image" width="50%"/> -->

---
# Installation

```bash
conda create -n glirel python=3.10 -y && conda activate glirel
cd GLiREL && pip install -e . && pip install -r requirements.txt
```

## To run experiments

```bash
# few_rel
cd data
python process_few_rel.py
cd ..
# adjust config
python train.py --config config_few_rel.yaml --log_dir logs-few-rel --relation_extraction
```

```bash
# wiki_zsl
cd data
curl -L -o wiki_all.json 'https://drive.google.com/uc?export=download&id=1ELFGUIYDClmh9GrEHjFYoE_VI1t2a5nK'
python process_wiki_zsl.py
cd ..
# adjust config
python train.py --config config_wiki_zsl.yaml --log_dir logs-wiki-zsl --relation_extraction

```

## Example training data

JSONL file:
```json
{
  "ner": [
    [7, 8, "Q4914513", "Binsey"], 
    [11, 13, "Q19686", "River Thames"]
  ], 
  "relations": [
    {
      "head": {"mention": "Binsey", "position": [7, 8], "type": "Q4914513"}, 
      "tail": {"mention": "River Thames", "position": [11, 13], "type": "Q19686"}, 
      "relation_id": "P206", 
      "relation_text": "located in or next to body of water"
    }
  ], 
  "tokenized_text": ["The", "race", "took", "place", "between", "Godstow", "and", "Binsey", "along", "the", "Upper", "River", "Thames", "."]
},
{
  "ner": [
    [9, 11, "Q4386693", "Legislative Assembly"], 
    [1, 4, "Q1848835", "Parliament of Victoria"]
  ], 
  "relations": [
    {
      "head": {"mention": "Legislative Assembly", "position": [9, 11], "type": "Q4386693"}, 
      "tail": {"mention": "Parliament of Victoria", "position": [1, 4], "type": "Q1848835"}, 
      "relation_id": "P361", 
      "relation_text": "part of"
    }
  ], 
  "tokenized_text": ["The", "Parliament", "of", "Victoria", "consists", "of", "the", "lower", "house", "Legislative", "Assembly", ",", "the", "upper", "house", "Legislative", "Council", "and", "the", "Queen", "of", "Australia", "."]
}


```



## Usage
Once you've downloaded the GLiREL library, you can import the `GLiREL` class. You can then load this model using `GLiREL.from_pretrained` and predict entities with `predict_relations`.

```python
from glirel import GLiREL
import spacy

model = GLiREL.from_pretrained("jackboyla/glirel_base")

text = "Jack Dorsey's father, Tim Dorsey, is a licensed pilot. Jack met his wife Sarah Paulson in New York in 2003. They have one son, Edward."

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

labels = ['country of origin', 'licensed to broadcast to', 'parent', 'followed by', 'located in or next to body of water', 'spouse', 'child']

tokens = [token.text for token in doc]

ner = [[ent.start, ent.end, ent.label_, ent.text] for ent in doc.ents]
print(f"Entities detected: {ner}")

relations = model.predict_relations(tokens, labels, threshold=0.01, ner=ner)

print('Number of relations:', len(relations))

sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
print("\nDescending Order by Score:")
for item in sorted_data_desc:
    print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | socre: {item['score']}")
```

### Expected Output

```
Entities detected: [[0, 2, 'PERSON', 'Jack Dorsey'], [5, 7, 'PERSON', 'Tim Dorsey'], [13, 14, 'PERSON', 'Jack'], [17, 19, 'PERSON', 'Sarah Paulson'], [20, 22, 'GPE', 'New York'], [23, 24, 'DATE', '2003'], [27, 28, 'CARDINAL', 'one'], [30, 31, 'PERSON', 'Edward']]
Number of relations: 90

Descending Order by Score:
['Sarah', 'Paulson'] --> spouse --> ['New', 'York'] | score: 0.6608812212944031
['Sarah', 'Paulson'] --> spouse --> ['Jack', 'Dorsey'] | score: 0.6601175665855408
['Edward'] --> spouse --> ['New', 'York'] | score: 0.6493653655052185
['one'] --> spouse --> ['New', 'York'] | score: 0.6480509042739868
['Edward'] --> spouse --> ['Jack', 'Dorsey'] | score: 0.6474933624267578
...
```

## Usage with spaCy (TBD)

You can also load GliREL into a regular spaCy NLP pipeline. Here's an example using a blank English pipeline, but you can use any spaCy model.

```python

```

### Expected Output

```

```

