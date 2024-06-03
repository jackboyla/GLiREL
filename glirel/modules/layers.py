import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import down_weight_loss


class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(LstmSeq2SeqEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)

        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output


# https://github.com/urchade/EnriCo/blob/main/modules/layers.py

def MLP(units, dropout, activation=nn.ReLU):
    # convert to integer
    units = [int(u) for u in units]

    assert len(units) >= 2
    layers = []
    for i in range(len(units) - 2):
        layers.append(nn.Linear(units[i], units[i + 1]))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(units[-2], units[-1]))
    return nn.Sequential(*layers)


class FilteringLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.filter_layer = nn.Linear(hidden_size, 2)

    def forward(self, embeds, label):

        # Extract dimensions
        B, num_spans, D = embeds.shape

        # Compute score using a predefined filtering function
        score = self.filter_layer(embeds)  # Shape: [B, num_spans, num_classes]

        # Modify label to binary (0 for negative class, 1 for positive)
        label_m = label.clone()
        label_m[label_m > 0] = 1

        # Initialize the loss
        filter_loss = 0
        if self.training:
            # Compute the loss if in training mode
            filter_loss = down_weight_loss(score.view(B * num_spans, -1),
                                           label_m.view(-1),
                                           sample_rate=0.,
                                           is_logit=True)

        # Compute the filter score (difference between positive and negative class scores)
        filter_score = score[..., 1] - score[..., 0]  # Shape: [B, num_spans]

        # Mask out filter scores for ignored labels
        filter_score = filter_score.masked_fill(label == -1, -1e9)

        if self.training:
            filter_score = filter_score.masked_fill(label_m > 0, 1e5)

        return filter_score, filter_loss


class CrossAtt(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAtt, self).__init__()

        assert d_model % num_heads == 0
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, add_zero_attn=True)

    def forward(self, query, kv, key_padding_mask=None):
        # query: [batch, seq_len, d_model]
        return self.mha(query, kv, kv, key_padding_mask=key_padding_mask)[0]


class ReadProcess(nn.Module):
    def __init__(self, d_model, num_heads, ffn_mul=4, dropout=0.1, read_only=False):
        super().__init__()

        self.read_only = read_only

        self.read = CrossAtt(d_model, num_heads)
        self.norm_read = nn.LayerNorm(d_model)

        if not read_only:
            self.process = CrossAtt(d_model, num_heads)
            self.norm_process = nn.LayerNorm(d_model)

        self.ffn = MLP([d_model, d_model * ffn_mul, d_model], dropout, activation=nn.GELU)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, input_embed, word_rep, input_mask, word_mask):
        # input: [batch, seq_len, d_model]
        input_mask = input_mask == False
        word_mask = word_mask == False

        # Cross-attention
        x = input_embed + self.read(self.norm_read(input_embed), word_rep, key_padding_mask=word_mask)

        # Self-attention
        if not self.read_only:
            x = x + self.process(self.norm_process(x), x, key_padding_mask=input_mask)

        return x + self.norm_ffn(self.ffn(x))


class RefineLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=1, ffn_mul=4, dropout=0.1, read_only=False):
        super().__init__()

        self.blocks = nn.ModuleList([ReadProcess(d_model, num_heads, ffn_mul, dropout, read_only)
                                     for _ in range(num_layers)])

        self.output = nn.Linear(d_model, d_model)

    def forward(self, input_embed, word_rep, input_mask, word_mask):
        for block in self.blocks:
            input_embed = block(input_embed, word_rep, input_mask, word_mask)
        return self.output(input_embed)


class ScorerLayer(nn.Module):
    def __init__(self, scoring_type="dot", hidden_size=768, dropout=0.1):
        super().__init__()

        self.scoring_type = scoring_type

        if scoring_type == "concat_proj":
            self.proj = MLP([hidden_size * 4, hidden_size * 4, 1], dropout)
        elif scoring_type == "dot_thresh":
            self.proj_thresh = MLP([hidden_size, hidden_size * 4, 2], dropout)
            self.proj_type = MLP([hidden_size, hidden_size * 4, hidden_size], dropout)
        elif scoring_type == "dot_norm":
            self.dy_bias_type = MLP([hidden_size, hidden_size * 4, 1], dropout)
            self.dy_bias_rel = MLP([hidden_size, hidden_size * 4, 1], dropout)
            self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, candidate_pair_rep, rel_type_rep):
        # candidate_pair_rep: [B, N, D]
        # rel_type_rep: [B, T, D]
        if self.scoring_type == "dot":
            return torch.einsum('BKD,BCD->BKC', candidate_pair_rep, rel_type_rep) # ([B, num_pairs, num_classes])

        elif self.scoring_type == "dot_thresh":
            # compute the scaling factor and threshold
            B, T, D = rel_type_rep.size()
            scaler = self.proj_thresh(rel_type_rep)  # [B, T, 2]
            # alpha: scaling factor, beta: threshold
            alpha, beta = scaler[..., 0].view(B, 1, T), scaler[..., 1].view(B, 1, T)
            alpha = F.softplus(alpha)  # reason: alpha should be positive
            # project the relation type representation
            rel_type_rep = self.proj_type(rel_type_rep)  # [B, T, D]
            # compute the score (before sigmoid)
            score = torch.einsum("bnd,btd->bnt", candidate_pair_rep, rel_type_rep)  # [B, N, T]
            return (score + beta) * alpha  # [B, N, T]

        elif self.scoring_type == "dot_norm":
            score = torch.einsum("bnd,btd->bnt", candidate_pair_rep, rel_type_rep)  # [B, N, T]
            bias_1 = self.dy_bias_type(rel_type_rep).transpose(1, 2)  # [B, 1, T]
            bias_2 = self.dy_bias_rel(candidate_pair_rep)  # [B, N, 1]
            return score + self.bias + bias_1 + bias_2

        elif self.scoring_type == "concat_proj":
            prod_features = candidate_pair_rep.unsqueeze(2) * rel_type_rep.unsqueeze(1)  # [B, N, T, D]
            diff_features = candidate_pair_rep.unsqueeze(2) - rel_type_rep.unsqueeze(1)  # [B, N, T, D]
            expanded_pair_rep = candidate_pair_rep.unsqueeze(2).repeat(1, 1, rel_type_rep.size(1), 1)
            expanded_rel_type_rep = rel_type_rep.unsqueeze(1).repeat(1, candidate_pair_rep.size(1), 1, 1)
            features = torch.cat([prod_features, diff_features, expanded_pair_rep, expanded_rel_type_rep],
                                 dim=-1)  # [B, N, T, 2D]
            return self.proj(features).squeeze(-1)