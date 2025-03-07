import pytest
import torch

from glirel.modules.deprecated_token_rep import TokenRepLayer as OriginalTokenRepLayer
from glirel.modules.token_rep import TokenRepLayer


@pytest.mark.parametrize(
    "model_name,fine_tune,subtoken_pooling,hidden_size,add_tokens",
    [
        ("microsoft/deberta-v3-small", False, "first", 768, []),
        ("microsoft/deberta-v3-small", True,  "mean", 768, ["[REL]", "[SEP]"]),
        ("microsoft/deberta-v3-small", False,  "last", 768, ["[REL]", "[SEP]"]),
        ("microsoft/deberta-v3-small", True,  "first_last", 768, ["[REL]", "[SEP]"]),
        ("microsoft/deberta-v3-large", True, "first", 1024, []),
        ("microsoft/deberta-v3-large", True, "first", 768, []),
        ("microsoft/deberta-v3-small", False, "first", 256, []),
        ("microsoft/deberta-v3-small", False, "first", 128, []),
    ],
)
def test_compare_original_and_modified(
    model_name, fine_tune, subtoken_pooling, hidden_size, add_tokens
):
    """
    Compare the output of the old (Flair-based) TokenRepLayer vs.
    the new TokenRepLayer for the same inputs.
    """

    # Instantiate both classes
    original_layer = OriginalTokenRepLayer(
        model_name=model_name,
        fine_tune=fine_tune,
        subtoken_pooling=subtoken_pooling,
        hidden_size=hidden_size,
        add_tokens=add_tokens,
    )
    new_layer = TokenRepLayer(
        model_name=model_name,
        fine_tune=fine_tune,
        subtoken_pooling=subtoken_pooling,
        hidden_size=hidden_size,
        add_tokens=add_tokens,
    )

    # Prepare a small batch of tokens
    tokens_batch = [
        ["Hello", "world"],  # length=2
        ["This", "is", "a", "test"],  # length=4
    ]
    lengths = torch.tensor([len(seq) for seq in tokens_batch])  # [2, 4]

    # Forward pass on original
    with torch.no_grad():
        original_out = original_layer(tokens_batch, lengths)

    # Forward pass on new
    with torch.no_grad():
        new_out = new_layer(tokens_batch, lengths)

    # Check keys
    assert "embeddings" in original_out, "Original output missing 'embeddings'"
    assert "mask" in original_out, "Original output missing 'mask'"
    assert "embeddings" in new_out, "New output missing 'embeddings'"
    assert "mask" in new_out, "New output missing 'mask'"

    # Compare shapes
    orig_emb = original_out["embeddings"]
    new_emb = new_out["embeddings"]
    orig_mask = original_out["mask"]
    new_mask = new_out["mask"]

    assert orig_emb.shape == new_emb.shape, (
        f"Embedding shape mismatch: original={orig_emb.shape}, new={new_emb.shape}"
    )
    assert orig_mask.shape == new_mask.shape, (
        f"Mask shape mismatch: original={orig_mask.shape}, new={new_mask.shape}"
    )

    torch.testing.assert_close(
        orig_emb, new_emb, rtol=1e-5, atol=1e-5,
        msg="Embeddings differ between original and modified versions!"
    )
    torch.testing.assert_close(
        orig_mask, new_mask, rtol=1e-5, atol=1e-5,
        msg="Mask differs between original and modified versions!"
    )


def test_compare_with_single_token_inputs():
    model_name = "microsoft/deberta-v3-small"
    hidden_size = 768
    original_layer = OriginalTokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=hidden_size,
        add_tokens=[]
    )
    new_layer = TokenRepLayer(
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

    with torch.no_grad():
        orig_out = original_layer(tokens_batch, lengths)
        new_out = new_layer(tokens_batch, lengths)

    # Check shapes
    orig_emb = orig_out["embeddings"]
    new_emb = new_out["embeddings"]
    assert orig_emb.shape == new_emb.shape
    torch.testing.assert_close(orig_emb, new_emb, rtol=1e-5, atol=1e-5)

    orig_mask = orig_out["mask"]
    new_mask = new_out["mask"]
    assert orig_mask.shape == new_mask.shape
    torch.testing.assert_close(orig_mask, new_mask, rtol=1e-5, atol=1e-5)


def test_compare_with_custom_tokens():
    model_name = "microsoft/deberta-v3-small"
    hidden_size = 768
    add_tokens = ["[CUSTOM_TOKEN]"]

    original_layer = OriginalTokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=hidden_size,
        add_tokens=add_tokens
    )
    new_layer = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first",
        hidden_size=hidden_size,
        add_tokens=add_tokens
    )

    tokens_batch = [
        ["Hello", "[CUSTOM_TOKEN]"],
        ["[CUSTOM_TOKEN]", "[CUSTOM_TOKEN]"],
    ]
    lengths = torch.tensor([2, 2])

    with torch.no_grad():
        orig_out = original_layer(tokens_batch, lengths)
        new_out = new_layer(tokens_batch, lengths)

    # Compare
    torch.testing.assert_close(
        orig_out["embeddings"], new_out["embeddings"], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        orig_out["mask"], new_out["mask"], rtol=1e-5, atol=1e-5
    )


def test_long_sequence():
    model_name = "microsoft/deberta-v3-small"
    hidden_size = 768

    original_layer = OriginalTokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first_last",
        hidden_size=hidden_size,
        add_tokens=[]
    )
    new_layer = TokenRepLayer(
        model_name=model_name,
        fine_tune=False,
        subtoken_pooling="first_last",
        hidden_size=hidden_size,
        add_tokens=[]
    )

    # Make a sequence of 1024 tokens
    tokens_batch = [["Token"] * 1024]
    lengths = torch.tensor([1024])

    with torch.no_grad():
        orig_out = original_layer(tokens_batch, lengths)
        new_out = new_layer(tokens_batch, lengths)

    torch.testing.assert_close(
        orig_out["embeddings"], new_out["embeddings"], rtol=1e-5, atol=1e-5
    )
