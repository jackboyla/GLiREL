import pytest
import torch

from glirel.modules.transformer_embeddings import (
    TransformerWordEmbeddings,
    fill_masked_elements,
    fill_mean_token_embeddings,
    insert_missing_embeddings
)
from glirel.modules.token_rep import MinimalSentence, TokenRepLayer


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

def test_compute_word_embedding_with_newline_characters():
    # newline characters are merged/removed in the deberta tokenizer
    # we must ensure that the token embeddings are correctly aligned with the original tokens despite this
    sentences = [
        ['[REL]', 'birthstone is', '[REL]', 'is related to', '[REL]', 'none', '[REL]', 'includes access', '[REL]', 'is a type of', '[REL]', 'biographers', '[REL]', 'secured place in', '[REL]', 'is the date of', '[SEP]', 'Third', ',', 'in', 'the', 'global', 'register', 'which', 'is', 'characterized', 'by', 'the', 'increased', 'pace', 'and', '\n ', 'scope', 'of', 'the', 'movement', 'of', 'people', ',', 'money', 'and', 'technology', ',', 'but', '\n ', 'also', 'ideals', 'of', 'human', 'solidarity', ',', 'images', 'of', 'human', 'possibility', '.', '\n ', 'Religion', 'is', 'a', 'transposable', 'term', 'replaced', 'by', 'other', 'terms', 'such', 'as', 'spirituality', 'but', '\n ', 'also', 'transferred', 'to', 'a', 'range', 'of', 'apparently', 'nonreligious', 'activities', ',', '\n ', 'such', 'as', 'sports', 'or', 'capitalism', '.', '\n ', 'So', ',', 'we', 'have', 'this', 'new', 'fluidity', 'with', 'the', 'very', 'usages', 'of', 'the', 'term', 'religion', ',', '\n ', 'The', 'global', 'register', 'has', 'also', 'produced', 'a', 'commodification', 'of', 'religion', '.', '\n ', 'In', 'these', 'transpositions', ',', 'religion', 'by', 'any', 'name', 'signifies', 'sensory', 'immediacy', ',', '\n ', 'promising', 'direct', 'experience', 'but', '\n ', 'also', 'technological', 'mediation', 'through', 'electronic', 'media', '.', '\n ', 'Global', 'religion', 'is', 'religion', 'in', 'motion', 'defined', 'by', 'mobility', ',', 'fluidity', ',', '\n ', 'and', 'circulations', ',', 'unanchored', 'from', 'any', 'stable', 'bedrock', 'whether', 'on', 'land', 'or', 'sea', '.', '\n ', 'Unanchored', 'from', 'any', 'necessary', 'relation', 'between', 'statements', 'of', ',', '\n ', 'unanchored', 'from', 'any', 'necessary', 'relation', 'between', 'statements', 'and', '\n ', 'facts', 'in', 'a', 'proliferating', 'swirl', 'of', 'signs', '.', '\n ', 'I', "'", 'm', 'not', 'sure', 'how', 'this', 'is', 'gon', 'na', 'work', ',', 'let', "'s", 'see', ',', 'shall', 'we', 'see', 'how', 'this', 'works', '?', '\n ', 'I', 'could', 'just', 'stop', ',', 'no', 'let', "'s", 'see', 'what', 'happens', '.', '\n ', 'Global', 'religion', ',', 'now', 'I', "'", 'm', 'going', 'into', 'the', 'future', 'here', ',', 'signaling', '\n ', 'the', 'future', 'of', 'religion', 'in', 'the', 'present', ',', 'is', 'a', 'mix', 'of', 'authenticity', 'and', 'fakery', '.', '\n ', 'In', 'a', 'book', 'on', 'religion', 'in', 'American', 'popular', 'culture', ',', 'I', 'argue', 'that', 'even', 'fakes', ',', 'frauds', ',', '\n ', 'and', 'charlatans', 'can', 'do', 'real', 'religious', 'work', '.', '\n ', 'By', 'religious', 'work', ',', 'I', 'mean', 'symbolic', 'labor', '.', '\n ', 'In', 'the', 'fields', 'and', 'factories', 'of', 'the', 'transcendent', 'and', '\n ', 'the', 'sacred', 'dealing', 'with', 'gods', 'and', 'ancestors', ',', '\n ', 'producing', 'sacred', 'objects', ',', 'sacred', 'times', ',', 'sacred', 'spaces', '.', '\n ', 'Thamsanqa', 'Jantjie', ',', 'the', 'fake', 'signer', 'at', 'the', 'Nelson', 'Mandela', 'Memorial', '\n ', 'of', 'December', '10th', ',', '2013', 'was', 'an', 'authentic', 'fake', '.', '\n ', 'Okay', ',', 'operating', 'in', 'the', 'realm', 'of', 'free', 'signifiers', '.', '\n'],
        ['[REL]', 'number of', '[REL]', 'has', '[REL]', 'documenting', '[REL]', 'replaced with', '[REL]', 'supports travel for', '[REL]', 'requires members from', '[REL]', 'composed of', '[SEP]', 'Registration', 'requires', 'adherence', 'to', 'minimum', 'standards', 'and', 'a', 'code', 'of', 'ethics', ',', 'as', 'well', 'as', 'the', 'fulfilment', 'of', 'certain', 'training', 'requirements', '.', '\n ', 'The', 'law', 'allows', 'members', 'of', 'the', 'Muslim', 'community', ',', 'irrespective', 'of', 'their', 'school', 'of', 'Islam', 'or', 'ethnicity', ',', 'to', 'have', 'personal', 'status', 'issues', 'governed', 'by', 'Islamic', 'law', ',', '“', 'as', 'varied', 'where', 'applicable', 'by', 'Malay', 'custom', '.', '”', 'Ordinarily', 'the', 'Shafi’i', 'school', 'of', 'law', 'is', 'used', ',', 'but', 'there', 'are', 'legal', 'provisions', 'for', 'use', 'of', '“', 'other', 'accepted', 'schools', 'of', 'Muslim', 'law', 'as', 'may', 'be', 'appropriate', '.', '”', 'Under', 'the', 'law', ',', 'a', 'sharia', 'court', 'has', 'exclusive', 'jurisdiction', 'over', 'marriage', 'issues', 'where', 'both', 'parties', 'are', 'or', 'were', 'married', 'as', 'Muslims', ',', 'including', 'divorce', ',', 'nullification', ',', 'or', 'judicial', 'separation', '.', 'The', 'sharia', 'court', 'has', 'concurrent', 'jurisdiction', 'with', 'the', 'family', 'court', 'and', 'family', 'division', 'of', 'the', 'high', 'court', 'over', 'disputes', 'related', 'to', 'custody', 'of', 'minors', 'and', 'disposition', 'of', 'property', 'upon', 'divorce', '.', 'The', 'President', 'of', 'the', 'country', 'appoints', 'the', 'president', 'of', 'the', 'sharia', 'court', '.', 'A', 'breach', 'of', 'a', 'sharia', 'court', 'order', 'is', 'a', 'criminal', 'offense', 'punishable', 'by', 'imprisonment', 'of', 'up', 'to', 'six', 'months', ',', 'and', 'an', 'individual', 'may', 'file', 'a', 'complaint', 'alleging', 'a', 'breach', 'in', 'the', 'family', 'justice', 'courts', '.', 'The', 'sharia', 'court', 'does', 'not', 'have', 'jurisdiction', 'over', 'personal', 'protection', 'orders', 'or', 'applications', 'for', 'maintenance', 'payments', ',', 'as', 'these', 'are', 'orders', 'issued', 'by', 'a', 'secular', 'family', 'court', '.', 'Appeals', 'within', 'the', 'sharia', 'system', 'go', 'to', 'an', 'appeals', 'board', 'also', 'in', 'the', 'sharia', 'system', 'that', 'is', 'composed', 'of', 'three', 'members', 'selected', 'by', 'the', 'president', 'of', 'the', 'MUIS', 'from', 'a', 'panel', 'of', 'at', 'least', 'seven', 'Muslims', 'nominated', 'every', 'three', 'years', 'by', 'the', 'President', 'of', 'the', 'country', '.', 'The', 'ruling', 'of', 'the', 'appeals', 'board', 'is', 'final', 'and', 'may', 'not', 'be', 'appealed', 'to', 'any', 'other', 'court', '.', '\n ', 'The', 'law', 'allows', 'Muslim', 'men', 'to', 'practice', 'polygamy', ',', 'but', 'the', 'Registry', 'of', 'Muslim', 'Marriages', 'may', 'refuse', 'requests', 'to', 'marry', 'additional', 'wives', 'after', 'soliciting', 'the', 'views', 'of', 'existing', 'wives', ',', 'reviewing', 'the', 'husband', '’s', 'financial', 'capability', ',', 'and', 'evaluating', 'his', 'ability', 'to', 'treat', 'the', 'wives', 'and', 'families', 'fairly', 'and', 'equitably', '.', 'By', 'law', ',', 'the', 'President', 'of', 'the', 'country', 'appoints', 'a', '“', 'male', 'Muslim', 'of', 'good', 'character', 'and', 'suitable', 'attainments', '”', 'as', 'the', 'Registrar', 'of', 'Muslim', 'Marriages', '.', '\n']
    ]
    hidden_dim = 768
    rep_layer = TokenRepLayer(
        model_name='microsoft/deberta-v3-small',
        fine_tune=True,
        subtoken_pooling="first",
        hidden_size=hidden_dim,
        add_tokens=['[REL]', '[SEP]', '[E]', '[/E]', '[E]', '[/E]']
    )
    token_embeddings = rep_layer.compute_word_embedding(sentences)
    assert token_embeddings.shape[0] == len(sentences)
    assert token_embeddings.shape[1] == max(len(s) for s in sentences)
    assert token_embeddings.shape[2] == hidden_dim
