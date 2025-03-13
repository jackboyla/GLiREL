import pytest
import torch

from glirel.modules.transformer_embeddings import (
    TransformerWordEmbeddings,
    fill_masked_elements,
    fill_mean_token_embeddings,
    insert_missing_embeddings
)
from glirel.modules.token_rep import MinimalSentence


def test_insert_missing_embeddings_empty():
    """
    If token_embeddings is empty but length_i>0, ensure we fill with zero-vectors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.zeros((0, 10), device=device)  # shape [0, hidden_dim=10]
    word_ids_i = torch.tensor([0, 1, 2], device=device)  # length_i=3
    out = insert_missing_embeddings(dummy, word_ids_i, length_i=3)
    assert out.shape == (3, 10), "Should fill with zeros for all tokens"
    assert torch.all(out == 0)


def test_insert_missing_embeddings_partial():
    """
    If token_embeddings has fewer rows than length_i, we insert zeros for missing positions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.rand((2, 4), device=device)  # e.g. shape [2, hidden_dim=4]
    word_ids_i = torch.tensor([0, 2, 2], device=device)  # length=3 => token idx 0,2,2
    # So token idx=1 was never found => we must insert zeros at row=1
    out = insert_missing_embeddings(dummy, word_ids_i, length_i=3)
    assert out.shape == (3, 4)
    # Check row 1 is zero
    assert torch.all(out[1] == 0)


def test_fill_masked_elements():
    """
    Basic shape check for fill_masked_elements.
    We'll set up a scenario with a first_mask or last_mask.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 5
    hidden_dim = 4
    # shape => (batch, seq_len, hidden_dim)
    hidden_states = torch.rand((batch_size, seq_len, hidden_dim), device=device)
    # Suppose word_ids => shape (batch, seq_len)
    word_ids = torch.tensor([
        [0, 0, 1, 2, 2],
        [0, 1, 1, 1, -100]
    ], device=device)
    lengths = torch.tensor([3, 3], device=device)  # only 3 tokens per sentence
    # Let's define a mask that picks the "first" subtoken
    # first_mask => row=0 => [True, False, True, True, False]
    # row=1 => e.g. [True, True, False, False, False]
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    mask[0, 0] = True
    mask[0, 2] = True
    mask[0, 3] = True
    mask[1, 0] = True
    mask[1, 1] = True

    # Prepare final embeddings => (batch, max_tokens=3, hidden_dim=4)
    all_token_embeddings = torch.zeros((batch_size, 3, hidden_dim), device=device)
    out = fill_masked_elements(all_token_embeddings, hidden_states, mask, word_ids, lengths)
    assert out.shape == (2, 3, 4)
    assert torch.allclose(out, all_token_embeddings)


def test_fill_mean_token_embeddings():
    bsz = 2
    max_tokens = 3
    emb_dim = 2

    all_token_embeddings = torch.zeros((bsz, max_tokens, emb_dim))

    # hidden_states: shape [batch_size, seq_len, emb_dim]
    hidden_states = torch.tensor([
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        [
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
            [50.0, 50.0],
        ]
    ])

    # word_ids: maps each subword index to its token index, or -1 for special tokens/ignored subwords
    word_ids = torch.tensor([
        [0, 0, 1, 2, -1],  # For sample 0, subwords (0,1) -> token 0, (2)-> token 1, (3)-> token 2, subword4 is ignored
        [0, 1, 1, -1, -1]  # For sample 1, subword0-> token 0, (1,2)-> token 1, subword(3,4) ignored
    ])

    token_lengths = torch.tensor([3, 2])  # Sample0 has 3 valid tokens, Sample1 has 2 valid tokens

    out = fill_mean_token_embeddings(all_token_embeddings, hidden_states, word_ids, token_lengths)

    # Manually compute the expected result for each sample:

    # Sample 0:
    # Token 0: subwords (0,1) => average of [ (1+2)/2, (1+2)/2 ] = [1.5, 1.5 ]
    # Token 1: subword (2)    => [3.0, 3.0]
    # Token 2: subword (3)    => [4.0, 4.0]
    # (Subword 4 is -1 => ignored)

    # Sample 1:
    # Token 0: subword (0) => [10.0, 10.0]
    # Token 1: subwords (1,2) => [ (20+30)/2, (20+30)/2 ] = [25.0, 25.0]
    # (Subwords (3,4) are -1 => ignored)
    # The third token (index 2) is beyond token_length=2 => zeros

    expected = torch.tensor([
        [[1.5,  1.5],
         [3.0,  3.0],
         [4.0,  4.0]],
        [[10.0, 10.0],
         [25.0, 25.0],
         [0.0,  0.0]]
    ])

    torch.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("subtoken_pooling", ["first", "last", "mean", "first_last"])
def test_custom_embedding_subtoken_pooling(subtoken_pooling):
    model_name = "microsoft/deberta-v3-small"
    custom_emb = TransformerWordEmbeddings(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling=subtoken_pooling,
        allow_long_sentences=True
    )

    sentences = [MinimalSentence(["Hello", "world!"]), MinimalSentence(["Subtoken", "test", "here"])]

    custom_emb.embed(sentences)
    # Check shape or existence of embeddings
    # e.g. the first sentence has 2 tokens => each token has an embedding of size embed_dim
    embed_dim = custom_emb.embedding_length
    for sent in sentences:
        for tok in sent.tokens:
            emb = tok.get_embedding(custom_emb.name)
            assert emb.shape[-1] == embed_dim


def test_custom_embedding_fine_tune():
    """
    Check that if fine_tune=False, requires_grad is False for all parameters
    """
    model_name = "microsoft/deberta-v3-small"
    custom_emb = TransformerWordEmbeddings(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first"
    )
    # Confirm no param requires grad
    for p in custom_emb.model.parameters():
        assert not p.requires_grad

    # If we set fine_tune=True, param should be True
    custom_emb2 = TransformerWordEmbeddings(
        model_name=model_name,
        fine_tune=True,
        subtoken_pooling="first"
    )
    # Confirm param requires grad
    found_trainable = any(p.requires_grad for p in custom_emb2.model.parameters())
    assert found_trainable, "Should have at least some trainable parameters with fine_tune=True"


def test_custom_embedding_empty_sentences():
    """
    If user passes empty list to embed, no error should occur
    """
    model_name = "microsoft/deberta-v3-small"
    custom_emb = TransformerWordEmbeddings(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first"
    )
    custom_emb.embed([])  # should do nothing gracefully


def test_custom_token():
    """
    Check we can add a custom token, resize, and see that it's recognized
    in the embedding matrix dimension and at runtime.
    """
    model_name = "microsoft/deberta-v3-small"
    custom_emb = TransformerWordEmbeddings(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first"
    )
    custom_emb.model.resize_token_embeddings(len(custom_emb.tokenizer))
    old_size = custom_emb.model.get_input_embeddings().weight.shape[0]

    custom_emb.tokenizer.add_tokens(["[CUSTOM]"])
    custom_emb.model.resize_token_embeddings(len(custom_emb.tokenizer))
    new_size = custom_emb.model.get_input_embeddings().weight.shape[0]

    assert new_size == old_size + 1, (
        f"Expected embedding matrix to grow from {old_size} to {old_size + 1}, but got {new_size}"
    )

    sentence = MinimalSentence(["Hello", "[CUSTOM]"])
    custom_emb.embed([sentence])

    # Check shape of final token embeddings
    for token in sentence.tokens:
        emb = token.get_embedding(custom_emb.name)
        assert emb.shape[-1] == custom_emb.embedding_length, (
            f"Token embedding dimension {emb.shape[-1]} != {custom_emb.embedding_length}"
        )
