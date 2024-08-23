from typing import List

import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from torch import nn
from torch.nn.utils.rnn import pad_sequence


# flair.cache_root = '/gpfswork/rech/pds/upa43yu/.cache'


class TokenRepLayer(nn.Module):
    def __init__(self, model_name: str = "bert-base-cased", fine_tune: bool = True, subtoken_pooling: str = "first",
                 hidden_size: int = 768,
                 add_tokens=["[SEP]", "[REL]"]
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

        B = lengths.size(0)
        max_length = lengths.max()
        mask = (torch.arange(max_length).view(1, -1).repeat(B, 1) < lengths.cpu().unsqueeze(1)).to(
            token_embeddings.device).long()
        return {"embeddings": token_embeddings, "mask": mask}

    def compute_word_embedding(self, tokens):
        sentences = [Sentence(i) for i in tokens]
        self.bert_layer.embed(sentences)
        token_embeddings = pad_sequence([torch.stack([t.embedding for t in k]) for k in sentences], batch_first=True)
        return token_embeddings
    

# https://github.com/urchade/GLiNER/blob/main/gliner/modeling/encoder.py
import torch
from torch import nn
from transformers import AutoModel, AutoConfig

#just wraping to allow to load previously created models
class Transformer(nn.Module):
    def __init__(self, config, from_pretrained):
        super().__init__()
        if from_pretrained:
            self.model = AutoModel.from_pretrained(config.model_name)
        else:
            if config.encoder_config is None:
                encoder_config = AutoConfig.from_pretrained(config.model_name)
                if config.vocab_size!=-1:
                    encoder_config.vocab_size = config.vocab_size
   
            else:
                encoder_config = config.encoder_config 
            self.model = AutoModel.from_config(encoder_config)
    
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output[0]
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False):
        super().__init__()

        self.bert_layer = Transformer( #transformer_model
            config, from_pretrained,
        )

        bert_hidden_size = self.bert_layer.model.config.hidden_size

        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, 
                                                                            pad_to_multiple_of)
    def forward(self, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.bert_layer(*args, **kwargs)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)

        return token_embeddings