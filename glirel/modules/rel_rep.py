import torch
from torch import nn
from glirel.modules.span_rep import create_projection_layer, SpanMarkerV0, extract_elements
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)




class SpanMarkerV1(nn.Module):
    """
    Marks and projects span endpoints using an MLP.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        # span_idx.shape    ([B, num_possible_spans, 2])

        start_rep = self.project_start(h)  # ([B, L, D])
        end_rep = self.project_end(h)      # ([B, L, D])

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])  # ([B, num_possible_spans, D])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])      # ([B, num_possible_spans, D])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()   # ([B, num_possible_spans, D*2])

        return self.out_project(cat) # ([B, number_of_entities, D])               #### .view(B, L, self.max_width, D)
    

def get_entity_pair_reps(entity_reps):
        B, num_entities, D = entity_reps.shape

        # Create a tensor [B, num_entities, num_entities, D] by repeating entity_reps for pairing
        # Expanding entity_reps to pair each with every other
        entity_reps_expanded = entity_reps.unsqueeze(2).expand(-1, -1, num_entities, -1)
        entity_reps_tiled = entity_reps.unsqueeze(1).expand(-1, num_entities, -1, -1)

        
        # Concatenate the representations of all possible pairs
        # The shape becomes [B, num_entities, num_entities, 2D]
        # NOTE: OOM error can occur here -- if there's too many entities
        pair_reps = torch.cat([entity_reps_expanded, entity_reps_tiled], dim=3)  # [B, num_entities, num_entities, 2 * D]

        # Now we have an entity pair matrix where each [i, j] element is the pair combination
        # of the i-th and j-th entities. We need to mask the diagonal (self-pairs).

        # Create a mask to exclude self-pairs
        indices = torch.arange(num_entities)
        mask = (indices.unsqueeze(0) != indices.unsqueeze(1))  # Create a mask to exclude self-pairs
        mask = mask.unsqueeze(0).expand(B, -1, -1)           # Expand mask for all batches

        combined_pairs = pair_reps[mask].view(B, -1, 2*D)    # Reshape to [B, num_valid_pairs, 2*D]

        
        return combined_pairs


class RelMarkerv0(nn.Module):
    """
    Marks and projects representations for all pairs of entities.
    """
    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.span_marker = SpanMarkerV1(hidden_size, max_width, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """
        h: torch.Tensor - The hidden states of the shape [batch_size, seq_len, hidden_size]
        span_idx: torch.Tensor - The span indices of entities of the shape [batch_size, num_entities, 2]
        """
        B, L, D = h.size()
        # [B, num_entities, 2]  -->  span_idx.size()
        entity_reps = self.span_marker(h, span_idx)  #  ([B, number_of_entities, D])  

        combined_pairs = get_entity_pair_reps(entity_reps)

        # combined_pairs is now a tensor of shape [batch_size, num_pairs, 2*hidden_size]
        # where num_pairs is num_entities * (num_entities - 1) / 2, the number of unique pairs.

        combined_pairs_out = self.out_project(combined_pairs)


        return combined_pairs_out


class RelRepLayer(nn.Module):
    """
    Various span representation approaches
    """

    def __init__(self, hidden_size, max_width, rel_mode, **kwargs):
        super().__init__()

        if rel_mode == 'marker':
            self.rel_rep_layer = RelMarkerv0(hidden_size, max_width, **kwargs)
        else:
            raise ValueError(f'Unknown rel mode {rel_mode}')

    def forward(self, x, *args):

        return self.rel_rep_layer(x, *args)
