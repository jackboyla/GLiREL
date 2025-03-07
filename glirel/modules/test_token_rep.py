import pytest

import torch
from torch import nn

from glirel.modules.token_rep import TokenRepLayer


@pytest.mark.parametrize(
    "model_name,fine_tune,subtoken_pooling,hidden_size,add_tokens",
    [
        ("microsoft/deberta-v3-large", True, "first", 1024, []),
        ("microsoft/deberta-v3-large", True, "first", 768, []),
        ("microsoft/deberta-v3-small", True,  "first", 768, []),
        ("microsoft/deberta-v3-small", False, "first", 256, []),
        ("microsoft/deberta-v3-small", True, "mean", 768, ["[REL]", "[SEP]"]),
        ("microsoft/deberta-v3-small", True, "last", 768, ["[REL]", "[SEP]"]),
        ("microsoft/deberta-v3-small", True, "first_last", 768, ["[REL]", "[SEP]"])
    ],
)
def test_token_rep_layer_shapes_and_mask(
    model_name, fine_tune, subtoken_pooling, hidden_size, add_tokens
):

    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=fine_tune,
        subtoken_pooling=subtoken_pooling,
        hidden_size=hidden_size,
        add_tokens=add_tokens,
    )

    tokens_batch = [
        ["Hello", "world"],                 # length=2
        ["This", "is", "a", "test"],        # length=4
    ]
    lengths = torch.tensor([len(seq) for seq in tokens_batch])  # => [2, 4]

    # Forward pass
    output = token_rep(tokens_batch, lengths)

    # Check we have embeddings and mask in the output
    assert "embeddings" in output and "mask" in output, "Output keys must include 'embeddings' and 'mask'."

    embeddings = output["embeddings"]
    mask = output["mask"]

    # Check shape of embeddings: (batch_size, max_length, hidden_size)
    assert embeddings.shape[0] == len(tokens_batch), "Batch size dim mismatch."
    assert embeddings.shape[1] == max(lengths), "Max length dim mismatch."
    assert embeddings.shape[2] == hidden_size, "Hidden size dim mismatch."

    # Check shape of mask: (batch_size, max_length)
    assert mask.shape == (len(tokens_batch), max(lengths)), "Mask shape mismatch."

    # Quick check that the mask is correct:
    # For example, for lengths=[2,4], the mask might be:
    #  tensor([[1,1,0,0],
    #          [1,1,1,1]])
    expected_mask = []
    max_len = max(lengths)
    for seq_len in lengths:
        row = [1]*seq_len.item() + [0]*(max_len - seq_len.item())
        expected_mask.append(row)
    expected_mask = torch.tensor(expected_mask, dtype=torch.long)
    torch.testing.assert_close(
        mask.cpu(), expected_mask, msg="Mask does not match expected positions of real vs. padded tokens."
    )


def test_token_rep_layer_projection():

    model_name = "microsoft/deberta-v3-small"
    new_hidden_size = 300  # something different than default
    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=new_hidden_size,
        add_tokens=[]
    )

    # We expect a 'projection' attribute
    assert hasattr(token_rep, "projection"), "Should have a projection layer if hidden_size != BERT hidden size."
    assert isinstance(token_rep.projection, nn.Linear), "Projection layer must be nn.Linear."

    # Forward pass with dummy data
    tokens_batch = [["hello"], ["this", "is", "test"]]
    lengths = torch.tensor([1, 3])
    output = token_rep(tokens_batch, lengths)
    emb = output["embeddings"]
    # Check last dimension is new_hidden_size
    assert emb.shape[-1] == new_hidden_size, "Embedding last dimension should match the projection size."


def test_token_rep_layer_no_projection():

    model_name = "microsoft/deberta-v3-small"
    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=768,
        add_tokens=[]
    )
    assert not hasattr(token_rep, "projection"), "Should NOT have a projection layer if hidden_size == BERT hidden size."


def test_token_rep_layer_empty_input():

    model_name = "microsoft/deberta-v3-small"
    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=768,
        add_tokens=[]
    )

    tokens_batch = []  # no sequences
    lengths = torch.tensor([], dtype=torch.long)

    # We assert that a RuntimeError occurs when the input is empty
    with pytest.raises(RuntimeError, match="received an empty list of sequences"):
        token_rep(tokens_batch, lengths)


def test_token_rep_layer_single_token_inputs():

    model_name = "microsoft/deberta-v3-small"
    hidden_size = 768
    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=hidden_size,
        add_tokens=[]
    )

    tokens_batch = [
        ["Hello"],
        ["World"],
        ["[REL]"],
    ]
    lengths = torch.tensor([1, 1, 1])

    output = token_rep(tokens_batch, lengths)
    emb = output["embeddings"]
    mask = output["mask"]

    # shape => (3, 1, hidden_size)
    assert emb.shape == (3, 1, hidden_size)
    # mask => (3, 1)
    assert mask.shape == (3, 1)
    # all mask entries should be 1 since length=1 for each
    assert torch.all(mask == 1)


def test_added_tokens_vocabulary():

    model_name = "microsoft/deberta-v3-small"
    add_tokens = ["[CUSTOM1]", "[CUSTOM2]"]
    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=768,
        add_tokens=add_tokens
    )
    tokenizer = token_rep.bert_layer.tokenizer

    for t in add_tokens:
        assert t in tokenizer.get_vocab(), f"Token {t} was not added to the vocabulary."


@pytest.mark.parametrize("subtoken_pooling", ["first", "last", "mean", "first_last"])
def test_subtoken_pooling_variations(subtoken_pooling):

    model_name = "microsoft/deberta-v3-small"
    token_rep = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling=subtoken_pooling,
        hidden_size=768,
        add_tokens=[]
    )

    tokens_batch = [
        ["Hello", "world!"],
        ["subtoken", "pooling", "test"],
    ]
    lengths = torch.tensor([2, 3])

    output = token_rep(tokens_batch, lengths)
    emb = output["embeddings"]
    mask = output["mask"]

    # shape => (2, 3, 768)
    assert emb.shape[0] == 2, "Batch dimension mismatch."
    assert emb.shape[1] == 3, "Max length dimension mismatch."
    assert emb.shape[2] == 768, "Embedding dimension mismatch for BERT base."
    assert mask.shape == (2, 3)
