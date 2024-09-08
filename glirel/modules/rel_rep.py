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
    

def get_entity_pair_reps(entity_reps, **kwargs):
    """
    Generates entity pair representations, constrained by the token distance between entities.
    
    Args:
        entity_reps: Tensor of shape [B, num_entities, D], representing the entity representations.
        span_idx: Tensor of shape [B, num_entities, 2], representing the start and end indices of each entity.
        max_distance: Integer, the maximum allowed token distance between entity pairs.
        
    Returns:
        combined_pairs: Tensor of shape [B, num_valid_pairs, 2 * D], representing the valid entity pair representations.
    """
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


def get_entity_pair_reps_v2(entity_reps, span_idx, max_distance=None):
    """
    Generates entity pair representations separately for coreference (unidirectional, without max distance constraint)
    and for relation classification (bidirectional, with max distance constraint).
    
    Args:
        entity_reps: Tensor of shape [B, num_entities, D], representing the entity representations.
        span_idx: Tensor of shape [B, num_entities, 2], representing the start and end indices of each entity.
        max_distance: Integer, the maximum allowed token distance for relation classification (None for coreference).
        
    Returns:
        coref_pairs: Tensor of shape [B, num_coref_pairs, 2 * D], representing the coreferent entity pair representations.
        relation_pairs: Tensor of shape [B, num_relation_pairs, 2 * D], representing the relation entity pair representations.
    """
    B, num_entities, D = entity_reps.shape

    # Get start indices of each entity span from span_idx (shape: [B, num_entities])
    start_idx = span_idx[:, :, 0]

    # Compute token distance between entity pairs
    start_idx_expanded = start_idx.unsqueeze(2).expand(-1, -1, num_entities)  # [B, num_entities, num_entities]
    start_idx_tiled = start_idx.unsqueeze(1).expand(-1, num_entities, -1)     # [B, num_entities, num_entities]
    token_distances = torch.abs(start_idx_expanded - start_idx_tiled)         # [B, num_entities, num_entities]

    # Create a mask to exclude self-pairs and ensure unidirectional pairs for coreference (i < j)
    indices = torch.arange(num_entities, device=entity_reps.device)
    self_pair_mask = indices.unsqueeze(0) < indices.unsqueeze(1)  # Unidirectional: Only keep (i, j) where i < j

    # Generate all valid entity pairs for coreference (no distance constraint, unidirectional)
    coref_mask = self_pair_mask.unsqueeze(0).expand(B, -1, -1)

    # Get the indices of valid coreference pairs
    valid_coref_pairs = torch.nonzero(coref_mask, as_tuple=False)  # Shape: [num_coref_pairs, 3] -> [batch_idx, i, j]

    # Extract the valid coreference entity representations using advanced indexing
    coref_batch_indices = valid_coref_pairs[:, 0]
    coref_i_indices = valid_coref_pairs[:, 1]
    coref_j_indices = valid_coref_pairs[:, 2]

    entity_reps_i_coref = entity_reps[coref_batch_indices, coref_i_indices]  # Shape: [num_coref_pairs, D]
    entity_reps_j_coref = entity_reps[coref_batch_indices, coref_j_indices]  # Shape: [num_coref_pairs, D]

    # Concatenate coreference entity representations
    coref_pairs = torch.cat([entity_reps_i_coref, entity_reps_j_coref], dim=1)  # Shape: [num_coref_pairs, 2 * D]

    # For relations, apply the token distance constraint (bidirectional)
    if max_distance is not None:
        distance_mask = token_distances <= max_distance  # Apply distance constraint
        relation_mask = self_pair_mask.unsqueeze(0).expand(B, -1, -1) | self_pair_mask.t().unsqueeze(0)  # Allow both (i, j) and (j, i)
        relation_mask = relation_mask & distance_mask  # Combine masks
    else:
        relation_mask = self_pair_mask.unsqueeze(0).expand(B, -1, -1)

    # Get the indices of valid relation pairs
    valid_relation_pairs = torch.nonzero(relation_mask, as_tuple=False)  # Shape: [num_relation_pairs, 3] -> [batch_idx, i, j]

    # Extract the valid relation entity representations using advanced indexing
    relation_batch_indices = valid_relation_pairs[:, 0]
    relation_i_indices = valid_relation_pairs[:, 1]
    relation_j_indices = valid_relation_pairs[:, 2]

    entity_reps_i_relation = entity_reps[relation_batch_indices, relation_i_indices]  # Shape: [num_relation_pairs, D]
    entity_reps_j_relation = entity_reps[relation_batch_indices, relation_j_indices]  # Shape: [num_relation_pairs, D]

    # Concatenate relation entity representations
    relation_pairs = torch.cat([entity_reps_i_relation, entity_reps_j_relation], dim=1)  # Shape: [num_relation_pairs, 2 * D]

    # Reshape coreference and relation pairs to [B, num_valid_pairs, 2 * D]
    coref_pairs = coref_pairs.view(B, -1, 2 * D)
    relation_pairs = relation_pairs.view(B, -1, 2 * D)

    return coref_pairs, relation_pairs



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
