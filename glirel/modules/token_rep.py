from typing import List

import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from torch import nn
from torch.nn.utils.rnn import pad_sequence


# flair.cache_root = '/gpfswork/rech/pds/upa43yu/.cache'


class TokenRepLayer(nn.Module):
    def __init__(self, model_name: str, fine_tune: bool, subtoken_pooling: str,
                 hidden_size: int,
                 add_tokens: List[str]
                 ):
        super().__init__()

        self.bert_layer = TransformerWordEmbeddings(
            model_name,
            fine_tune=fine_tune,
            subtoken_pooling=subtoken_pooling,
            allow_long_sentences=True
        )

        # add tokens to vocabulary
        self.bert_layer.tokenizer.add_tokens(add_tokens)

        # resize token embeddings
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
        mask = (torch.arange(max_length).view(1, -1).repeat(B, 1) < lengths.cpu().unsqueeze(1)).to(
            token_embeddings.device).long()
        return {"embeddings": token_embeddings, "mask": mask}

    def compute_word_embedding(self, tokens):
        sentences = [Sentence(i) for i in tokens]
        """
        sentences[0]
            Sentence[29]: "[REL] doctoral advisor [REL] publisher [REL] connecting line [SEP] The Church of the Faroe Islands ( Faroese Fólkakirkjan ) is one of the smallest of the worlds state church es ."
        [t for t in sentences[0].tokens]
            [Token[0]: "[REL]", Token[1]: "doctoral advisor", Token[2]: "[REL]", Token[3]: "publisher", Token[4]: "[REL]", Token[5]: "connecting line", Token[6]: "[SEP]", Token[7]: "The", Token[8]: "Church", Token[9]: "of", Token[10]: "the", Token[11]: "Faroe", Token[12]: "Islands", Token[13]: "(", Token[14]: "Faroese", Token[15]: "Fólkakirkjan", Token[16]: ")", Token[17]: "is", Token[18]: "one", Token[19]: "of", Token[20]: "the", Token[21]: "smallest", Token[22]: "of", Token[23]: "the", Token[24]: "worlds", Token[25]: "state", Token[26]: "church", Token[27]: "es", Token[28]: "."]
        """
        self.bert_layer.embed(sentences)
        token_embeddings = pad_sequence([torch.stack([t.embedding for t in k]) for k in sentences], batch_first=True)
        return token_embeddings
