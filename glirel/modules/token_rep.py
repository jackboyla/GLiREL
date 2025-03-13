from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from glirel.modules.transformer_embeddings import TransformerWordEmbeddings


class MinimalToken:
    def __init__(self, text: str):
        self.text = text
        self._embeddings = {}

    def set_embedding(self, name: str, vector: torch.Tensor):
        self._embeddings[name] = vector

    def get_embedding(self, name: str) -> torch.Tensor:
        return self._embeddings[name]


class MinimalSentence:
    """
    A minimal drop-in replacement for Flair's Sentence
    """
    def __init__(self, list_of_words: List[str]):
        self.tokens = [MinimalToken(w) for w in list_of_words]


class TokenRepLayer(nn.Module):
    def __init__(self, model_name: str, fine_tune: bool, subtoken_pooling: str,
                 hidden_size: int, add_tokens: List[str]):
        super().__init__()

        self.bert_layer = TransformerWordEmbeddings(
            model_name,
            fine_tune=fine_tune,
            subtoken_pooling=subtoken_pooling,
            allow_long_sentences=True
        )

        # Flair’s code automatically inserts "[FLERT]" if use_context_separator=True that is the default
        # Adding also it here to replicate the same behavior
        if "[FLERT]" not in self.bert_layer.tokenizer.get_vocab():
            self.bert_layer.tokenizer.add_special_tokens({"additional_special_tokens": ["[FLERT]"]})

        # Add tokens to vocabulary
        if add_tokens:
            self.bert_layer.tokenizer.add_tokens(add_tokens)

        # Resize token embeddings
        self.bert_layer.model.resize_token_embeddings(len(self.bert_layer.tokenizer))

        bert_hidden_size = self.bert_layer.embedding_length

        if hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, hidden_size)

    def forward(self, tokens: List[List[str]], lengths: torch.Tensor):
        token_embeddings = self.compute_word_embedding(tokens)

        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)

        B = len(lengths)
        max_length = lengths.max()
        mask = (torch.arange(max_length).view(1, -1).repeat(B, 1)
                < lengths.cpu().unsqueeze(1)).to(token_embeddings.device).long()

        return {"embeddings": token_embeddings, "mask": mask}

    def compute_word_embedding(self, tokens):
        sentences = [MinimalSentence(toks) for toks in tokens]
        """
        sentences[0]
            Sentence[29]: "[REL] doctoral advisor [REL] publisher [REL] connecting line [SEP] The Church of the Faroe Islands ( Faroese Fólkakirkjan ) is one of the smallest of the worlds state church es ."
        [t for t in sentences[0].tokens]
            [Token[0]: "[REL]", Token[1]: "doctoral advisor", Token[2]: "[REL]", Token[3]: "publisher", Token[4]: "[REL]", Token[5]: "connecting line", Token[6]: "[SEP]", Token[7]: "The", Token[8]: "Church", Token[9]: "of", Token[10]: "the", Token[11]: "Faroe", Token[12]: "Islands", Token[13]: "(", Token[14]: "Faroese", Token[15]: "Fólkakirkjan", Token[16]: ")", Token[17]: "is", Token[18]: "one", Token[19]: "of", Token[20]: "the", Token[21]: "smallest", Token[22]: "of", Token[23]: "the", Token[24]: "worlds", Token[25]: "state", Token[26]: "church", Token[27]: "es", Token[28]: "."]
        """
        self.bert_layer.embed(sentences)
        token_embeddings = pad_sequence(
            [torch.stack([tok.get_embedding(self.bert_layer.name) for tok in s.tokens])
             for s in sentences],
            batch_first=True
        )
        return token_embeddings
