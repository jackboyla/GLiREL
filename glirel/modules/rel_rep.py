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
    
class SpanMarkerV2(nn.Module):
    """
    Efficiently computes span representations by pooling over embeddings.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.projection = create_projection_layer(hidden_size, dropout)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, D] - Hidden states from the encoder.
        span_idx: [B, num_spans, 2] - Start and end indices of spans.
        Returns:
            span_reps: [B, num_spans, D] - Pooled span representations.
        """
        B, L, D = h.size()
        _, num_spans, _ = span_idx.size()

        # Compute lengths of each span
        lengths = span_idx[:, :, 1] - span_idx[:, :, 0] + 1  # [B, num_spans]
        max_span_length = lengths.max().item()

        # Create position offsets
        position_offsets = torch.arange(max_span_length, device=h.device).unsqueeze(0).unsqueeze(0)  # [1, 1, max_span_length]

        # Compute positions
        start_positions = span_idx[:, :, 0].unsqueeze(-1)  # [B, num_spans, 1]
        positions = start_positions + position_offsets  # [B, num_spans, max_span_length]

        # Create mask for valid positions
        span_mask = (position_offsets < lengths.unsqueeze(-1))  # [B, num_spans, max_span_length]

        # Clamp positions to valid range
        positions = positions * span_mask.long()
        positions = positions.clamp(0, L - 1)  # Ensure indices are within [0, L - 1]

        # Flatten positions for gathering
        positions_flat = positions.view(B, -1)  # [B, num_spans * max_span_length]

        # Gather embeddings
        h_expanded = h.unsqueeze(1).expand(-1, num_spans, -1, -1)  # [B, num_spans, L, D]
        h_expanded = h_expanded.contiguous().view(B * num_spans, L, D)
        positions_flat = positions_flat.view(B * num_spans, -1)
        gathered_embeddings = h_expanded.gather(1, positions_flat.unsqueeze(-1).expand(-1, -1, D))  # [B * num_spans, max_span_length, D]

        # Reshape to [B, num_spans, max_span_length, D]
        gathered_embeddings = gathered_embeddings.view(B, num_spans, max_span_length, D)

        # Apply mask
        span_mask = span_mask.float().unsqueeze(-1)  # [B, num_spans, max_span_length, 1]
        sum_embeddings = (gathered_embeddings * span_mask).sum(dim=2)  # [B, num_spans, D]
        span_lengths = lengths.float().unsqueeze(-1)  # [B, num_spans, 1]

        # Compute mean embeddings
        span_reps = sum_embeddings / span_lengths  # [B, num_spans, D]

        # Apply projection layer
        span_reps = self.projection(span_reps)  # [B, num_spans, D]

        return span_reps


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


def get_entity_pair_reps_v2(entity_reps, span_idx, relations_idx):
    """
    Generates entity pair representations based on the given relation indices.
    
    Args:
        entity_reps: Tensor of shape [B, num_entities, D], representing the entity representations.
        span_idx: Tensor of shape [B, num_entities, 2], representing the start and end indices of each entity.
        relations_idx: Tensor of shape [B, num_relations, 2, 2], where the last two indices 
                      are the start and end indices of an entity pair (e.g. [[0, 4], [6, 8]]).
        
    Returns:
        relation_pairs: Tensor of shape [B, num_relations, 2 * D], representing the relation 
                        entity pair representations.
    """
    B, num_entities, D = entity_reps.shape
    _, num_relations, _, _ = relations_idx.shape

    # Extract start and end token indices for head and tail from relations_idx
    head_start_tokens = relations_idx[:, :, 0, 0]  # [B, num_relations]
    head_end_tokens = relations_idx[:, :, 0, 1]  # [B, num_relations]
    tail_start_tokens = relations_idx[:, :, 1, 0]  # [B, num_relations]
    tail_end_tokens = relations_idx[:, :, 1, 1]  # [B, num_relations]

    # Extract start and end token indices for each entity in span_idx
    entity_starts = span_idx[:, :, 0]  # [B, num_entities]
    entity_ends = span_idx[:, :, 1]    # [B, num_entities]

    # Create masks to identify which entity each token belongs to
    head_mask = (head_start_tokens.unsqueeze(2) == entity_starts.unsqueeze(1)) & (head_end_tokens.unsqueeze(2) == entity_ends.unsqueeze(1))  # [B, num_relations, num_entities]
    tail_mask = (tail_start_tokens.unsqueeze(2) == entity_starts.unsqueeze(1)) & (tail_end_tokens.unsqueeze(2) == entity_ends.unsqueeze(1))  # [B, num_relations, num_entities]

    # Use argmax to get the indices of the entities that match the tokens in head and tail
    head_entity_indices = torch.argmax(head_mask.int(), dim=2)  # [B, num_relations]
    tail_entity_indices = torch.argmax(tail_mask.int(), dim=2)  # [B, num_relations]

    # Gather the entity representations using the found indices
    head_entity_reps = torch.gather(entity_reps, 1, head_entity_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, num_relations, D]
    tail_entity_reps = torch.gather(entity_reps, 1, tail_entity_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, num_relations, D]

    # Concatenate head and tail representations for each relation
    relation_pairs = torch.cat([head_entity_reps, tail_entity_reps], dim=-1)  # [B, num_relations, 2 * D]

    return relation_pairs


class RelMarkerv0(nn.Module):
    """
    Marks and projects representations for all pairs of entities.
    """
    def __init__(self, span_mode: str, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        if span_mode == 'markerv1':
            self.span_marker = SpanMarkerV1(hidden_size, max_width, dropout)
        elif span_mode == 'markerv2':
            self.span_marker = SpanMarkerV2(hidden_size, max_width, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor, relations_idx: torch.Tensor = None) -> torch.Tensor:
        """
        h: torch.Tensor - The hidden states of the shape [batch_size, seq_len, hidden_size]
        span_idx: torch.Tensor - The span indices of entities of the shape [batch_size, num_entities, 2]
        """
        B, L, D = h.size()
        # [B, num_entities, 2]  -->  span_idx.size()
        entity_reps = self.span_marker(h, span_idx)  #  ([B, number_of_entities, D])  

        combined_pairs = get_entity_pair_reps_v2(entity_reps, span_idx=span_idx, relations_idx=relations_idx)
        # combined_pairs = get_entity_pair_reps(entity_reps)

        # combined_pairs is now a tensor of shape [batch_size, num_pairs, 2*hidden_size]
        # where num_pairs is num_entities * (num_entities - 1) / 2, the number of unique pairs.

        combined_pairs_out = self.out_project(combined_pairs)


        return combined_pairs_out


class RelRepLayer(nn.Module):
    """
    Various span representation approaches
    """

    def __init__(self, hidden_size, max_width, rel_mode, span_mode, **kwargs):
        super().__init__()

        if rel_mode == 'marker':
            self.rel_rep_layer = RelMarkerv0(span_mode, hidden_size, max_width, **kwargs)
        else:
            raise ValueError(f'Unknown rel mode {rel_mode}')

    def forward(self, x, *args, **kwargs):

        return self.rel_rep_layer(x, *args, **kwargs)
