import pytest
from glirel import GLiREL

@pytest.fixture
def setup_glirel():
    model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    return model

def test_glirel_predict_with_ground_truth(setup_glirel):
    model = setup_glirel

    tokens = [
        "The", "race", "took", "place", "between", "Godstow", "and",
        "Binsey", "along", "the", "Upper", "River", "Thames", "."
    ]
    labels = [
        'country of origin',
        'located in or next to body of water',
        'licensed to broadcast to',
        'father',
        'followed by',
        'characters',
    ]
    ner = [
        [7, 7, "Q4914513", "Binsey"], 
        [11, 12, "Q19686", "River Thames"]
    ]
    ground_truth_relations = [
        {
          "head": {"mention": "Binsey", "position": [7, 7], "type": "LOC"},
          "tail": {"mention": "River Thames", "position": [11, 12], "type": "Q19686"},
          "relation_text": "located in or next to body of water"
        }
    ]

    relations, loss = model.predict_relations(
        tokens,
        labels,
        threshold=0.0,
        ner=ner,
        top_k=-1,
        ground_truth_relations=ground_truth_relations
    )

    assert isinstance(relations, list), "Expected relations to be a list."
    assert len(relations) > 0, "No relations returned by predict_relations."
    assert loss is not None, "Loss should not be None."

def test_glirel_predict_with_ground_truth_incorrect_position(setup_glirel):
    model = setup_glirel

    tokens = [
        "The", "race", "took", "place", "between", "Godstow", "and",
        "Binsey", "along", "the", "Upper", "River", "Thames", "."
    ]
    labels = [
        'country of origin',
        'located in or next to body of water',
        'licensed to broadcast to',
        'father',
        'followed by',
        'characters',
    ]
    ner = [
        [7, 7, "Q4914513", "Binsey"], 
        [11, 12, "Q19686", "River Thames"]
    ]
    bad_ground_truth_relations = [
        {
          "head": {"mention": "Binsey", "position": [7, 20], "type": "LOC"}, # incorrect position
          "tail": {"mention": "River Thames", "position": [11, 12], "type": "Q19686"},
          "relation_text": "located in or next to body of water"
        }
    ]

    with pytest.raises(AssertionError, match=".*position.*"):
        relations, loss = model.predict_relations(
            tokens,
            labels,
            threshold=0.0,
            ner=ner,
            top_k=-1,
            ground_truth_relations=bad_ground_truth_relations
        )

def test_glirel_predict_without_ground_truth(setup_glirel):
    model = setup_glirel

    tokens = [
        "The", "race", "took", "place", "between", "Godstow", "and",
        "Binsey", "along", "the", "Upper", "River", "Thames", "."
    ]
    labels = [
        'country of origin',
        'located in or next to body of water',
        'licensed to broadcast to',
        'father',
        'followed by',
        'characters',
    ]
    ner = [
        [7, 7, "Q4914513", "Binsey"], 
        [11, 12, "Q19686", "River Thames"]
    ]


    relations = model.predict_relations(
        tokens,
        labels,
        threshold=0.0,
        ner=ner,
        top_k=-1,
    )

    assert isinstance(relations, list), "Expected relations to be a list."