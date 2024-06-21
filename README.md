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
pip install glirel
```

## Usage
Once you've downloaded the GLiREL library, you can import the `GLiREL` class. You can then load this model using `GLiREL.from_pretrained` and predict entities with `predict_relations`.

```python
from glirel import GLiREL
import spacy

model = GLiREL.from_pretrained("jackboyla/glirel_base")

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

text = 'Derren Nesbitt had a history of being cast in "Doctor Who", having played villainous warlord Tegana in the 1964 First Doctor serial "Marco Polo".'

labels = ['country of origin', 'licensed to broadcast to', 'father', 'followed by', 'characters']

ner = [[26, 27, 'Q2989881', 'Marco Polo'], [22, 23, 'Q2989412', 'First Doctor']]

relations = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=1)

print('Number of relations:', len(relations))

sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
print("\nDescending Order by Score:")
for item in sorted_data_desc:
    print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | socre: {item['score']}")
```

### Expected Output

```
Number of relations: 2

Descending Order by Score:
{'head_pos': [26, 28], 'tail_pos': [22, 24], 'head_text': ['Marco', 'Polo'], 'tail_text': ['First', 'Doctor'], 'label': 'characters', 'score': 0.9923334121704102}
{'head_pos': [22, 24], 'tail_pos': [26, 28], 'head_text': ['First', 'Doctor'], 'tail_text': ['Marco', 'Polo'], 'label': 'characters', 'score': 0.9915636777877808}
```

## Constrain labels
In practice, we usually want to define the types of entities that can exist as a head and/or tail of a relationship. This is already implemented in GLiREL:

```python
labels = {"glirel_labels": {
    'co-founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]}, 
    'no relation': {},  # head and tail can be any entity type 
    'country of origin': {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["LOC", "GPE"]}, 
    'parent': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]}, 
    'located in or next to body of water': {"allowed_head": ["LOC", "GPE", "FAC"], "allowed_tail": ["LOC", "GPE"]},  
    'spouse': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
    'child': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
    'founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},  
    'founded on date': {"allowed_head": ["ORG"], "allowed_tail": ["DATE"]},
    'headquartered in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},  
    'acquired by': {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},  
    'subsidiary of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]}, 
    }
}
```

## Usage with spaCy

You can also load GliREL into a regular spaCy NLP pipeline. Here's an example using an English pipeline.

```python
import spacy
import glirel

# Load a blank spaCy model or an existing one
nlp = spacy.load('en_core_web_sm')

# Add the GLiREL component to the pipeline
nlp.add_pipe("glirel", after="ner")

# Now you can use the pipeline with the GLiREL component
text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California."

labels = {"glirel_labels": {
    'co-founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]}, 
    'country of origin': {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["LOC", "GPE"]}, 
    'licensed to broadcast to': {"allowed_head": ["ORG"]},  
    'no relation': {},  
    'parent': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]}, 
    'followed by': {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["PERSON", "ORG"]},  
    'located in or next to body of water': {"allowed_head": ["LOC", "GPE", "FAC"], "allowed_tail": ["LOC", "GPE"]},  
    'spouse': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
    'child': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
    'founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},  
    'headquartered in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},  
    'acquired by': {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},  
    'subsidiary of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]}, 
    }
}

# Add the labels to the pipeline at inference time
docs = list( nlp.pipe([(text, labels)], as_tuples=True) )
relations = docs[0][0]._.relations

print('Number of relations:', len(relations))

sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
print("\nDescending Order by Score:")
for item in sorted_data_desc:
    print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | score: {item['score']}")

```

### Expected Output

```
Number of relations: 5

Descending Order by Score:
['Apple', 'Inc.'] --> headquartered in --> ['California'] | score: 0.9854260683059692
['Apple', 'Inc.'] --> headquartered in --> ['Cupertino'] | score: 0.9569844603538513
['Steve', 'Wozniak'] --> co-founder --> ['Apple', 'Inc.'] | score: 0.09025496244430542
['Steve', 'Jobs'] --> co-founder --> ['Apple', 'Inc.'] | score: 0.08805803954601288
['Ronald', 'Wayne'] --> co-founder --> ['Apple', 'Inc.'] | score: 0.07996643334627151
```


## To run experiments

FewRel: ~56k examples
WikiZSL: ~85k examples

```bash
# few_rel
cd data
python process_few_rel.py
cd ..
# adjust config
python train.py --config config_few_rel.yaml
```

```bash
# wiki_zsl
cd data
python process_wiki_zsl.py
cd ..
# <adjust config>
python train.py --config config_wiki_zsl.yaml
```

## Example training data

NOTE that the entity indices are inclusive i.e `"Binsey"` is `[7, 7]`. This differs from spaCy where the end index is exclusive (in this case spaCy would set the indices to `[7, 8]`)

JSONL file:
```json
{
  "ner": [
    [7, 7, "Q4914513", "Binsey"], 
    [11, 12, "Q19686", "River Thames"]
  ], 
  "relations": [
    {
      "head": {"mention": "Binsey", "position": [7, 7], "type": "Q4914513"},
      "tail": {"mention": "River Thames", "position": [11, 12], "type": "Q19686"}, 
      "relation_text": "located in or next to body of water"
    }
  ], 
  "tokenized_text": ["The", "race", "took", "place", "between", "Godstow", "and", "Binsey", "along", "the", "Upper", "River", "Thames", "."]
},
{
  "ner": [
    [9, 10, "Q4386693", "Legislative Assembly"], 
    [1, 3, "Q1848835", "Parliament of Victoria"]
  ], 
  "relations": [
    {
      "head": {"mention": "Legislative Assembly", "position": [9, 10], "type": "Q4386693"}, 
      "tail": {"mention": "Parliament of Victoria", "position": [1, 3], "type": "Q1848835"}, 
      "relation_text": "part of"
    }
  ], 
  "tokenized_text": ["The", "Parliament", "of", "Victoria", "consists", "of", "the", "lower", "house", "Legislative", "Assembly", ",", "the", "upper", "house", "Legislative", "Council", "and", "the", "Queen", "of", "Australia", "."]
}


```





