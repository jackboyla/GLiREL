import pytest
import spacy
from unittest.mock import patch, MagicMock

@pytest.fixture
def setup_nlp():
    # Load spaCy pipeline and add GLiREL component
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "glirel", 
        after="ner",
        config={"model": "jackboyla/glirel-large-v0", "batch_size": 1, "threshold": 0.0}
    )
    return nlp

def test_glirel_pipeline(setup_nlp):
    text = (
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. "
        "The company is headquartered in Cupertino, California."
    )
    labels = {
        "glirel_labels": {
            "co-founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
            "headquartered in": {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},
        }
    }

    # Run pipeline
    docs = list(setup_nlp.pipe([(text, labels)], as_tuples=True))
    doc, context = docs[0]

    # Check relations in the output
    assert hasattr(doc._, 'relations'), "Expected relations attribute in the output."
    relations = doc._.relations
    assert len(relations) > 0, "Expected > 0 relations in the output."

