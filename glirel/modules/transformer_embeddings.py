import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def fill_masked_elements(all_token_embeddings, hidden_states, mask, word_ids, lengths):
    """
    For 'first' or 'last' subtoken pooling: copy out exactly the subtoken embeddings
    that match the mask + are valid word_ids, and place them in the correct positions.
    """
    batch_size = all_token_embeddings.size(0)
    for i in range(batch_size):
        keep = hidden_states[i][mask[i] & (word_ids[i] >= 0)]
        replaced = insert_missing_embeddings(keep, word_ids[i], lengths[i])
        all_token_embeddings[i, : lengths[i], :] = replaced
    return all_token_embeddings


def insert_missing_embeddings(token_embeddings, word_ids_i, length_i):
    """
    If some token indices [0..length_i-1] never appeared in 'token_embeddings',
    insert zero-vectors at those positions.
    """
    if token_embeddings.size(0) == 0:
        # No subtokens found at all, so fill with zeros
        return torch.zeros(
            int(length_i),
            token_embeddings.size(-1),
            device=token_embeddings.device,
            dtype=token_embeddings.dtype
        )
    elif token_embeddings.size(0) < length_i:
        # Potentially insert zero-vectors for any missing token positions
        for idx in range(int(length_i)):
            if not (word_ids_i == idx).any():
                zero_vec = torch.zeros_like(token_embeddings[:1])
                token_embeddings = torch.cat(
                    (token_embeddings[:idx], zero_vec, token_embeddings[idx:]),
                    dim=0
                )
    return token_embeddings


def fill_mean_token_embeddings(all_token_embeddings, hidden_states, word_ids, token_lengths):
    """
    For 'mean' subtoken pooling: sum all subtoken embeddings for each token ID
    and divide by the subtoken count.
    """
    _, max_tokens, emb_dim = all_token_embeddings.shape
    # mask to ignore special tokens (CLS, SEP, or None)
    mask = (word_ids >= 0)

    # sum embeddings for each (batch, token_id)
    all_token_embeddings.scatter_add_(
        dim=1,
        index=word_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, emb_dim),
        src=hidden_states * mask.unsqueeze(-1).float(),
    )

    # count how many subtokens contributed per token
    subtoken_counts = torch.zeros_like(all_token_embeddings[:, :, 0])
    subtoken_counts.scatter_add_(
        1,
        word_ids.clamp(min=0),
        mask.float()
    )

    # average
    all_token_embeddings = torch.where(
        subtoken_counts.unsqueeze(-1) > 0,
        all_token_embeddings / subtoken_counts.unsqueeze(-1),
        torch.zeros_like(all_token_embeddings),
    )

    # zero out positions beyond the actual token length
    max_len = max_tokens
    idx_range = torch.arange(max_len, device=token_lengths.device).unsqueeze(0)
    valid_mask = (idx_range < token_lengths.unsqueeze(1))
    all_token_embeddings = all_token_embeddings * valid_mask.unsqueeze(-1)

    return all_token_embeddings


class TransformerWordEmbeddings(nn.Module):
    """
    A drop-in replacement for flair's `TransformerWordEmbeddings`:
    """
    def __init__(self, model_name: str, fine_tune: bool, subtoken_pooling: str, allow_long_sentences: bool = True):
        """
        :param model_name: Hugging Face model ID or path
        :param fine_tune: Whether to keep the model parameters trainable
        :param subtoken_pooling: 'first', 'last', 'mean', or 'first_last'
        :param allow_long_sentences: Whether to truncate long sentences
        """
        super().__init__()
        self.name = f"TransformerWordEmbeddings({model_name})"
        self.model_name = model_name
        self.fine_tune = fine_tune
        self.subtoken_pooling = subtoken_pooling
        self.allow_long_sentences = allow_long_sentences

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze or unfreeze
        if not fine_tune:
            for p in self.model.parameters():
                p.requires_grad = False

        # If we do 'first_last', dimension doubles
        hidden_size = self.model.config.hidden_size
        if subtoken_pooling == "first_last":
            self._embedding_length = hidden_size * 2
        else:
            self._embedding_length = hidden_size

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def embed(self, sentences):
        """
        Expects a list of "sentence-like" objects.
        Each "sentence" must have a `.tokens` list.
        Each "token" must have:
          - a .text attribute
          - a .set_embedding(name, vector) method
        """
        if not sentences:
            return

        # Prepare input: list of list-of-strings
        batch_of_lists = []
        for s in sentences:
            # s.tokens is a list of tokens, each with .text
            batch_of_lists.append([t.text for t in s.tokens])

        # Tokenize
        encoding = self.tokenizer(
            batch_of_lists,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=not self.allow_long_sentences
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        device = next(self.model.parameters()).device  # move to same device as model
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        # shape = [batch_size, seq_len, hidden_dim]
        last_hidden = outputs.last_hidden_state

        batch_size, seq_len, hidden_dim = last_hidden.shape

        # Reconstruct which subtoken belongs to which token
        # word_ids(i) => a list of length seq_len with an integer or None
        word_ids_batch = []
        max_token_count = 0
        for i in range(batch_size):
            w_ids = encoding.word_ids(batch_index=i)
            if w_ids is None:
                # fallback (slow tokenizer) => all None
                w_ids = [None]*seq_len
            # figure out how many tokens are in that sample
            valid_ids = [x for x in w_ids if x is not None]
            if valid_ids:
                max_id = max(valid_ids)
                token_count = max_id + 1
            else:
                token_count = 0
            if token_count > max_token_count:
                max_token_count = token_count
            word_ids_batch.append(w_ids)

        # Build a [batch_size, seq_len] tensor of token IDs, or -100 if None
        word_ids_tensor = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        for i in range(batch_size):
            for j, w_id in enumerate(word_ids_batch[i]):
                if w_id is not None:
                    word_ids_tensor[i, j] = w_id

        # Token lengths per sentence
        token_lengths = []
        for i in range(batch_size):
            valid = [x for x in word_ids_batch[i] if x is not None]
            token_lengths.append((max(valid)+1) if valid else 0)
        token_lengths_tensor = torch.tensor(token_lengths, device=device, dtype=torch.long)

        # Prepare final [batch_size, max_token_count, embedding_dim]
        embed_dim = self.embedding_length
        all_token_embeddings = torch.zeros(
            (batch_size, max_token_count, embed_dim),
            device=device, dtype=last_hidden.dtype
        )

        # Subtoken pooling
        true_tensor = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

        if self.subtoken_pooling == "first":
            # 'first' subtoken => mask out the beginning of each word
            gain_mask = (word_ids_tensor[:, 1:] != word_ids_tensor[:, :-1])
            # first position is always True
            first_mask = torch.cat([true_tensor, gain_mask], dim=1)
            fill_masked_elements(all_token_embeddings, last_hidden, first_mask, word_ids_tensor, token_lengths_tensor)

        elif self.subtoken_pooling == "last":
            # 'last' subtoken => mask out the boundary at the next subtoken
            gain_mask = (word_ids_tensor[:, 1:] != word_ids_tensor[:, :-1])
            last_mask = torch.cat([gain_mask, true_tensor], dim=1)
            fill_masked_elements(all_token_embeddings, last_hidden, last_mask, word_ids_tensor, token_lengths_tensor)

        elif self.subtoken_pooling == "first_last":
            gain_mask = word_ids_tensor[:, 1:] != word_ids_tensor[:, :-1]
            first_mask = torch.cat([true_tensor, gain_mask], dim=1)
            last_mask = torch.cat([gain_mask, true_tensor], dim=1)

            # Fill the first half
            all_token_embeddings[:, :, :hidden_dim] = fill_masked_elements(
                all_token_embeddings[:, :, :hidden_dim],
                last_hidden,
                first_mask,
                word_ids_tensor,
                token_lengths_tensor
            )
            # Fill the second half
            all_token_embeddings[:, :, hidden_dim:] = fill_masked_elements(
                all_token_embeddings[:, :, hidden_dim:],
                last_hidden,
                last_mask,
                word_ids_tensor,
                token_lengths_tensor
            )

        elif self.subtoken_pooling == "mean":
            fill_mean_token_embeddings(all_token_embeddings, last_hidden, word_ids_tensor, token_lengths_tensor)
        else:
            raise ValueError(f"Unknown subtoken_pooling={self.subtoken_pooling}")

        # Now store each token's embedding
        # For each sample i
        for i, sentence in enumerate(sentences):
            length_i = token_lengths[i]
            # slice out the relevant portion
            embs_i = all_token_embeddings[i, :length_i]  # shape [length_i, embed_dim]
            # set embedding on each token
            for token_idx, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, embs_i[token_idx])

    def __str__(self):
        return self.name
