from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
import os
import logging


logger = logging.getLogger(__name__)


def generate_entity_pairs_indices(span_idx):
    num_entities = span_idx.size(0)  # [num_ents, 2]

    # Expand and tile to create all possible pairs
    span_idx_expanded = span_idx.unsqueeze(1).expand(-1, num_entities, -1)  #  ([num_entities, num_entities, 2])
    span_idx_tiled = span_idx.unsqueeze(0).expand(num_entities, -1, -1)     #  ([num_entities, num_entities, 2])

    # we now need a mask to exclude self-pairs
    indices = torch.arange(num_entities)
    indices_expanded = indices.unsqueeze(1).expand(-1, num_entities)
    indices_tiled = indices.unsqueeze(0).expand(num_entities, -1)
    # Create a mask to filter out self-pairs
    self_pair_mask = indices_expanded != indices_tiled

    # Apply the mask to filter out self-pairs
    span_idx_expanded_filtered = span_idx_expanded[self_pair_mask]  #  ([num_unique_pairs, 2])
    span_idx_tiled_filtered = span_idx_tiled[self_pair_mask]        #  ([num_unique_pairs, 2])


    # Stack the pairs back in shape [num_pairs, 2, 2]
    combined_pairs = torch.stack((span_idx_expanded_filtered, span_idx_tiled_filtered), dim=1)

    return combined_pairs  #  ([num_unique_pairs, 2 ->start_index, 2 ->end_index])


class InstructBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_width = config.max_width
        self.base_config = config

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
            elif (tail_idx, head_idx) in rel_label_dict:
                # assign the same label as the reverse relation (if it exists)
                label = rel_label_dict[(head_idx, tail_idx)]
                rel_labels.append(label)
            else:
                rel_labels.append(0)

        return rel_labels
    

    def preprocess_spans(self, tokens, ner, classes_to_id, relations):

        max_len = self.base_config.max_len

        if len(tokens) > max_len:
            logger.warn(f"Token length {len(tokens)} is longer than max length {max_len}. Truncating.")
            seq_length = max_len
            tokens = tokens[:max_len]
        else:
            seq_length = len(tokens)

        spans_idx = []


        # TODO: test this!!
        for ner_span in ner:
            start, end = ner_span[0], ner_span[1]
            spans_idx.append((start, end))

        MAX_SPANS = 35     # max number of span pairs -- can be increased with more VRAM
        if len(spans_idx) > MAX_SPANS:
            spans_idx = spans_idx[: MAX_SPANS]
            logger.warn(f"Truncating relations and ner spans because there are too many (> {MAX_SPANS})")
        spans_idx_list = spans_idx
        
        spans_idx = torch.LongTensor(spans_idx)                   # [num_possible_spans, 2]
        relations_idx = generate_entity_pairs_indices(spans_idx)  # [num_ent_pairs, 2, 2]

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
            rel_label = torch.LongTensor(self.get_rel_labels(relations_idx, rel_label_dict, classes_to_id))  # [num_ent_pairs]

        else:  # no labels --> predict
            rel_label_dict = defaultdict(int)
            rel_label = torch.LongTensor([rel_label_dict[i] for i in relations_idx])


        # mask for valid spans
        valid_span_mask = spans_idx[:, 1] > seq_length - 1

        # mask invalid positions
        span_label = torch.ones(spans_idx.size(0), dtype=torch.long)
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

    def collate_fn(self, batch_list, entity_types=None, train_relation_types=None):
        # batch_list: list of dict containing tokens, ner
        if entity_types is None:
            assert train_relation_types is not None, "`train_relation_types` must be provided for relation extraction data loader"
            negs = self.get_negatives_rel(train_relation_types, 100)
            class_to_ids = []
            id_to_classes = []
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

                # # add "no relation" to labels
                # types = [NO_RELATION_STR] + types

                if len(types) < self.base_config.num_train_rel_types:
                    logger.warn(f"Relation types less than num_train_rel_types: {len(types)} < {self.base_config.num_train_rel_types}")


                # shuffle (every epoch)
                random.shuffle(types)

                # random drop
                if len(types) != 0 and self.base_config.random_drop:
                    num_rels = random.randint(1, len(types))
                    types = types[ :num_rels]

                types = types[ : 50] # self.base_config.num_train_rel_types


                # supervised training
                if "label" in b:
                    types = sorted(b["label"])

                class_to_id = {k: v for v, k in enumerate(types, start=1)}
                id_to_class = {k: v for v, k in class_to_id.items()}
                class_to_ids.append(class_to_id)
                id_to_classes.append(id_to_class)

            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids[i], b.get('relations')) for i, b in enumerate(batch_list)
            ]

        else:
            class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
            id_to_classes = {k: v for v, k in class_to_ids.items()}
            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids, b.get('relations')) for b in batch_list
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
        

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_label != -1,
            'span_label': span_label,
            'rel_label': rel_label if 'rel_label' in locals() else None,
            'relations_idx': [el['relations_idx'] for el in batch],
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

    def create_dataloader(self, data, entity_types=None, train_relation_types=None, **kwargs):
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types, train_relation_types), **kwargs)

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
