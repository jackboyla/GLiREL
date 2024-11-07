from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
import os
import logging
from copy import copy, deepcopy


logger = logging.getLogger(__name__)

def insert_entity_markers(tokens, span_idx, relations, entity_start_token, entity_end_token):
    """
    tokens: List[str] - tokenized input sequence
    span_idx: List[Tuple[int, int]] - list of (start_idx, end_idx) for entities
    relations: List[Dict] - list of relation dictionaries
    entity_start_token: str - entity start token
    entity_end_token: str - entity end token

    ['<e>', 'John', 'Doe', '</e>', 'is', 'a', 'software', 'engineer', 'at', '<e>', 'Google', '</e>', 'in', '<e>', 'Dublin', '</e>', '.']
    """
    if entity_start_token in tokens or entity_end_token in tokens:
        # logger.warning(f"Entity markers already present in tokens. Skipping insertion.")
        return tokens, span_idx, relations
    prev_tokens = copy(tokens)
    prev_relations = deepcopy(relations)
    offset = 0
    adjusted_span_idx = []
    span2adjusted_span = {}
    for start, end in sorted(span_idx, key=lambda x: x[0]):
        adjusted_start = start + offset
        tokens.insert(start + offset, entity_start_token)
        offset += 1
        tokens.insert(end + offset + 1, entity_end_token)
        offset += 1
        adjusted_end = end + offset
        adjusted_span_idx.append((adjusted_start, adjusted_end))
        span2adjusted_span[(start, end)] = (adjusted_start, adjusted_end)

    adjusted_relations = []
    for rel in relations:
        head_idx = tuple(rel['head']['position'])
        tail_idx = tuple(rel['tail']['position'])
        try:
            adjusted_head_idx = span2adjusted_span[head_idx]
        except:
            import ipdb; ipdb.set_trace()
        try:
            adjusted_tail_idx = span2adjusted_span[tail_idx]
        except:
            import ipdb; ipdb.set_trace()
        rel['head']['position'] = list(adjusted_head_idx)
        rel['tail']['position'] = list(adjusted_tail_idx)
        adjusted_relations.append(rel)
    
    adjusted_relations = sorted(adjusted_relations, key=lambda x: x['head']['position'][0])
    return tokens, adjusted_span_idx, adjusted_relations

def generate_entity_pairs_indices(span_idx, max_distance: int | None = None, device: str = 'cpu'):
    """
    Generates a combined entity pair indices tensor for both coreference and relation classification.
    Coreference pairs are unidirectional (i < j), and relation pairs are bidirectional (i ≠ j),
    with a distance constraint for relation pairs if specified.

    Args:
        span_idx: Tensor of shape [num_entities, 2], representing the start and end indices of each entity.
        max_distance: Integer, the maximum allowed token distance for relation classification (None for coreference).
        
    Returns:
        combined_pairs: Tensor of shape [num_pairs, 2, 2], representing start/end indices for all valid entity pairs.
    """
    num_entities = span_idx.size(0)  # [num_ents, 2]

    # Expand and tile to create all possible pairs
    span_idx_expanded = span_idx.unsqueeze(1).expand(-1, num_entities, -1)  # ([num_entities, num_entities, 2])
    span_idx_tiled = span_idx.unsqueeze(0).expand(num_entities, -1, -1)     # ([num_entities, num_entities, 2])

    # Indices for self-pair exclusion and directionality
    indices = torch.arange(num_entities, device=device)
    indices_expanded = indices.unsqueeze(1).expand(-1, num_entities)
    indices_tiled = indices.unsqueeze(0).expand(num_entities, -1)

    # Coreference: unidirectional pairs (i < j)
    coref_mask = indices_expanded < indices_tiled  # Keep only pairs where i < j

    # Relation pairs: bidirectional pairs, excluding self-pairs (i != j)
    relation_mask = indices_expanded != indices_tiled

    # Compute token distances between entity pairs (based on start indices)
    start_idx_expanded = span_idx[:, 0].unsqueeze(1).expand(-1, num_entities)  # [num_entities, num_entities]
    start_idx_tiled = span_idx[:, 0].unsqueeze(0).expand(num_entities, -1)     # [num_entities, num_entities]
    token_distances = torch.abs(start_idx_expanded - start_idx_tiled)  # [num_entities, num_entities]

    # Apply distance constraint for relation pairs
    if max_distance is not None:
        distance_mask = token_distances <= max_distance
        relation_mask = relation_mask & distance_mask

    # Combine coreference and relation masks
    combined_mask = coref_mask | relation_mask  # Union of coref and relation masks

    # Apply the combined mask to filter pairs
    span_idx_filtered_expanded = span_idx_expanded[combined_mask]
    span_idx_filtered_tiled = span_idx_tiled[combined_mask]

    # Stack the pairs back into shape [num_pairs, 2, 2]
    combined_pairs = torch.stack((span_idx_filtered_expanded, span_idx_filtered_tiled), dim=1)

    return combined_pairs  # ([num_pairs, 2, 2])


class InstructBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_width = config.max_width
        self.base_config = config
        self.base_config.coreference_label = getattr(config, "coreference_label", "SELF")  # NOTE: this label is given a special index to denote coreference (i.e -2)
        self.max_entity_pair_distance = config.max_entity_pair_distance
        self.device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.base_config.entity_start_token, self.base_config.entity_end_token = "[E]", "[/E]"
        if self.base_config.span_marker_mode == 'markerv2':
            logger.info(f"Using SpanMarkerV2. Adding entity markers: {self.base_config.entity_start_token}, {self.base_config.entity_end_token}")

    def get_dict(self, spans, classes_to_id):
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag
    
    def get_rel_dict(self, relations, classes_to_id):
        dict_tag = defaultdict(int)
        for rel in relations:
            if rel['relation_text'] in classes_to_id:
                dict_tag[(tuple(rel['head']['position']), tuple(rel['tail']['position']))] = classes_to_id[rel['relation_text']]
        return dict_tag
    
    def get_rel_labels(self, relations_idx, rel_label_dict, classes_to_id):
        # get the class for each relation pair

        relations_idx = relations_idx.tolist()

        # assign label if in dataset else 0
        rel_labels = []
        for rel in relations_idx:
            head_idx, tail_idx = tuple(rel[0]), tuple(rel[1])
            if (head_idx, tail_idx) in rel_label_dict:
                label = rel_label_dict[(head_idx, tail_idx)]
                rel_labels.append(label)
            # elif (tail_idx, head_idx) in rel_label_dict:
            #     # assign the same label as the reverse relation (if it exists)
            #     label = rel_label_dict[(head_idx, tail_idx)]
            #     rel_labels.append(label)
            else:
                rel_labels.append(0)

        return rel_labels
    

    def preprocess_spans(self, tokens, ner, classes_to_id, relations):

        max_len = self.base_config.max_len

        if len(tokens) > max_len:
            logger.warning(f"Token length {len(tokens)} is longer than max length {max_len}. Truncating.")
            seq_length = max_len
            tokens = tokens[:max_len]
        else:
            seq_length = len(tokens)

        spans_idx = []


        for ner_span in ner:
            start, end = ner_span[0], ner_span[1]
            spans_idx.append((start, end))

        if hasattr(self.base_config, "add_entity_markers") and self.base_config.add_entity_markers:
            tokens, spans_idx, relations = insert_entity_markers(
                tokens, spans_idx, relations, 
                self.base_config.entity_start_token, self.base_config.entity_end_token
            )
            seq_length = len(tokens)

        # MAX_SPANS = 35     # NOTE: max number of span pairs -- can be increased with more VRAM
        # if len(spans_idx) > MAX_SPANS:
        #     logger.warn(f"Truncating relations and ner spans because there are too many ({len(spans_idx)} > {MAX_SPANS})")
        #     spans_idx = spans_idx[: MAX_SPANS]
        spans_idx_list = spans_idx
              
        spans_idx = torch.tensor(spans_idx, dtype=torch.long, device=self.device)  # [num_possible_spans, 2]
        relations_idx = generate_entity_pairs_indices(
            spans_idx, max_distance=self.base_config.max_entity_pair_distance,
            device=self.device
        )  # [num_ent_pairs, 2, 2]

        if relations is not None:  # training
            included_relations = []
            # if we need to truncate the number of relations
            for rel in relations:
                head_idx = (rel['head']['position'][0], rel['head']['position'][1]) 
                tail_idx = (rel['tail']['position'][0], rel['tail']['position'][1]) 
                if head_idx in spans_idx_list and tail_idx in spans_idx_list:
                    included_relations.append(rel)

            relations = included_relations

            # get the class for each relation pair
            rel_label_dict = self.get_rel_dict(relations, classes_to_id)
            # 0 for null labels
            rel_label = torch.tensor(
                self.get_rel_labels(relations_idx, rel_label_dict, classes_to_id),
                dtype=torch.long, device=self.device
            )  # [num_ent_pairs]

        else:  # no labels --> predict
            rel_label_dict = defaultdict(int)
            rel_label = torch.tensor([rel_label_dict[i] for i in relations_idx], dtype=torch.long, device=self.device)


        # mask for valid spans
        valid_span_mask = spans_idx[:, 1] > seq_length - 1

        # mask invalid positions
        span_label = torch.ones(spans_idx.size(0), dtype=torch.long, device=self.device)
        span_label = span_label.masked_fill(valid_span_mask, -1)  # [num_possible_spans]

        # mask for valid relations
        valid_rel_mask = relations_idx[:, 1, 1] > seq_length - 1
        rel_label = rel_label.masked_fill(valid_rel_mask, -1)


        out = {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': seq_length,
            'entities': ner,
            'relations': relations if 'relations' in locals() else None,
            'rel_label': rel_label if 'rel_label' in locals() else None,
            'relations_idx': relations_idx if 'relations_idx' in locals() else None,
        }
        return out

    def collate_fn(self, batch_list, relation_types=None, train_relation_types=None, device=None):
        if device:
            self.device = torch.device(device)
        assert hasattr(self.base_config, "fixed_relation_types"), "`fixed_relation_types` must be set in config"
        class_to_ids = []
        id_to_classes = []

        def _substitute_coref_label(coref_label, class_to_ids):
            """
            Assigns a special index to the coreference label. 
            This is used to distinguish coreference from other relation types if we decide to compute coreference loss separately.
            """
            if isinstance(coref_label, str):
                coref_label = [coref_label.lower()]
            else:
                coref_label = [l.lower() for l in coref_label]
            for key in class_to_ids.keys():
                if key.lower() in coref_label:
                    class_to_ids[key] = class_to_ids[key] * -50

            return class_to_ids

        # batch_list: list of dict containing tokens, ner
        if relation_types is None:
            # training
            self.processing_mode = "training"
            assert train_relation_types is not None, "`train_relation_types` must be provided for relation extraction data loader"
            negs = self.get_negatives_rel(train_relation_types, 100)
            for b in batch_list:
                # negs = b["negative"]
                random.shuffle(negs)


                positive_types = list(set([el['relation_text'] for el in b['relations']]))

                # make up to num_train_rel_types using as many negatives as needed (none if there's already enough positives)
                remainder_relations = max(0, int(self.base_config.num_train_rel_types) - len(positive_types))
                # remainder_relations -= 1  # save space for "no_relation"
                negs_i = [negative for negative in negs if negative not in positive_types][:remainder_relations]

                # this is the list of all possible relation types (positive and negative)
                types = list(set(positive_types + negs_i))

                # shuffle (every epoch)
                if getattr(self.base_config, "shuffle_types", True):
                    random.shuffle(types)

                # random drop
                if len(types) != 0 and self.base_config.random_drop:
                    num_rels = random.randint(1, len(types))
                    types = types[ :num_rels]

                types = types[ : self.base_config.num_train_rel_types]


                # supervised training
                if "label" in b:
                    types = sorted(b["label"])

                class_to_id = {k: v for v, k in enumerate(types, start=1)}
                class_to_id = _substitute_coref_label(self.base_config.coreference_label, class_to_id)
                id_to_class = {k: v for v, k in class_to_id.items()}
                class_to_ids.append(class_to_id)
                id_to_classes.append(id_to_class)

            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids[i], b.get('relations')) for i, b in enumerate(batch_list)
            ]

        else:
            # evaluation
            self.processing_mode = "evaluation"
            if (self.base_config.fixed_relation_types is True):
                # relation labels are fixed across all batches, e.g for evaluating m=15, etc
                class_to_id = {k: v for v, k in enumerate(relation_types, start=1)}
                # class_to_id = _substitute_coref_label(self.base_config.coreference_label, class_to_id)  # NOTE: change COREFERENCE LABEL TO -2
                id_to_class = {k: v for v, k in class_to_id.items()}
                class_to_ids = [class_to_id] * len(batch_list)
                id_to_classes = [id_to_class] * len(batch_list)
            else:
                # relation labels are different for each batch
                for i, b in enumerate(batch_list):
                    if 'relations' in b:
                        # eval during training
                        instance_relation_types = list(set([el['relation_text'] for el in b['relations']]))
                    else:
                        # provided batch of label lists in the wild
                        instance_relation_types = list(set([r for r in relation_types[i]]))
                    class_to_id = {k: v for v, k in enumerate(instance_relation_types, start=1)}
                    # class_to_id = _substitute_coref_label(self.base_config.coreference_label, class_to_id)  # NOTE: change COREFERENCE LABEL TO -2
                    id_to_class = {k: v for v, k in class_to_id.items()}
                    class_to_ids.append(class_to_id)
                    id_to_classes.append(id_to_class)
                # logger.info(f"Number of eval relation types per instance: {[len(d) for d in class_to_ids]}")

            
            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids[i], b.get('relations')) for i, b in enumerate(batch_list)
            ]

        span_idx = pad_sequence(
            [b['span_idx'] for b in batch], batch_first=True, padding_value=0
        )

        span_label = pad_sequence(
            [el['span_label'] for el in batch], batch_first=True, padding_value=-1
        )

        rel_label = pad_sequence(
            [el['rel_label'] for el in batch], batch_first=True, padding_value=-1
        )

        relations_idx = pad_sequence(
            [el['relations_idx'] for el in batch], batch_first=True, padding_value=-1
        )

        return {
            'seq_length': torch.tensor([el['seq_length'] for el in batch], dtype=torch.long, device=self.device),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_label != -1,
            'span_label': span_label,
            'rel_label': rel_label if 'rel_label' in locals() else None,
            'relations_idx': relations_idx,
            'entities': [el['entities'] for el in batch],
            'relations': [el.get('relations') for el in batch],
            'classes_to_id': class_to_ids,
            'id_to_classes': id_to_classes,
        }

    @staticmethod
    def get_negatives(batch_list, sampled_neg=5):
        ent_types = []
        for b in batch_list:
            types = set([el[2] for el in b['ner']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        # sample negatives
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]
    
    @staticmethod
    def get_negatives_rel(train_relation_types, sampled_neg=100):
        # sample negatives
        random.shuffle(train_relation_types)
        return train_relation_types[:sampled_neg]

    def create_dataloader(self, data, relation_types=None, train_relation_types=None, **kwargs):
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, relation_types, train_relation_types), **kwargs)

    def set_sampling_params(self, max_types, shuffle_types, random_drop, max_neg_type_ratio, max_len, num_train_rel_types=None):
        """
        Sets sampling parameters on the given model.

        Parameters:
        - model: The model object to update.
        - max_types: Maximum types parameter.
        - shuffle_types: Boolean indicating whether to shuffle types.
        - random_drop: Boolean indicating whether to randomly drop elements.
        - max_neg_type_ratio: Maximum negative type ratio.
        - max_len: Maximum length parameter.
        """
        self.base_config.max_types = max_types
        self.base_config.shuffle_types = shuffle_types
        self.base_config.random_drop = random_drop
        self.base_config.max_neg_type_ratio = max_neg_type_ratio
        self.base_config.max_len = max_len
        self.base_config.num_train_rel_types = num_train_rel_types
