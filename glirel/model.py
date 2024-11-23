import argparse
import json
from pathlib import Path
import re
import os
from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import time
from tqdm import tqdm
import glirel
from glirel.modules.layers import LstmSeq2SeqEncoder, ScorerLayer, FilteringLayer, RefineLayer
from glirel.modules.base import InstructBase
from glirel.modules.evaluator import greedy_search, RelEvaluator
from glirel.modules.span_rep import SpanRepLayer
from glirel.modules.rel_rep import RelRepLayer
from glirel.modules.token_rep import TokenRepLayer
from glirel.modules import loss_functions
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from typing import List, Dict, Union
from loguru import logger


class GLiREL(InstructBase, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # [REL] token
        self.rel_token = "[REL]"
        self.sep_token = "[SEP]"

        label_embed_strategies = ['ent_token', 'label', 'both']
        assert self.config.label_embed_strategy in label_embed_strategies, f"Invalid label_embed_strategy: {self.config.label_embed_strategy}. Must be one of {label_embed_strategies}"

        self.positive_weight = getattr(self.config, 'positive_weight', 2.0)  # weight for positive labels (Default: 2.0)
        self.negative_weight = getattr(self.config, 'negative_weight', 1.0)  # weight for negative labels (Default: 1.0)

        self.threshold_search_metric = getattr(self.config, 'threshold_search_metric', 'micro_f1')  # metric to use for threshold search (Default: 'micro_f1')

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        # add_tokens = [self.rel_token, self.sep_token]
        add_tokens = [self.rel_token, self.sep_token, self.base_config.entity_start_token, self.base_config.entity_end_token]
        if hasattr(self.base_config, "entity_start_token"):
            add_tokens += self.base_config.entity_start_token, self.base_config.entity_end_token
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=add_tokens)

        # hierarchical representation of tokens (zaratiana et al, 2022)
        # https://arxiv.org/pdf/2203.14710.pdf
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )


        self.span_rep_layer = RelRepLayer(
            rel_mode=config.rel_mode,
            span_mode=config.span_marker_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout,
        )

        # prompt representation (FFN)
        self.prompt_rep_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

        # refine relation representation
        if hasattr(config, "refine_relation") and config.refine_relation:
            self.refine_relation = RefineLayer(
                config.hidden_size, config.hidden_size // 64, num_layers=1, ffn_mul=config.ffn_mul,
                dropout=config.dropout,
                read_only=True
            )

        # refine prompt representation
        if hasattr(config, "refine_prompt") and config.refine_prompt:
            self.refine_prompt = RefineLayer(
                config.hidden_size, config.hidden_size // 64, num_layers=2, ffn_mul=config.ffn_mul,
                dropout=config.dropout,
                read_only=True
            )


        # coreference resolution
        if getattr(config, "coref_classifier", False):
            self.coref_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )


        # scoring layer
        self.scorer = ScorerLayer(config.scorer, hidden_size=config.hidden_size, dropout=config.dropout)


    def get_optimizer(self, lr_encoder, lr_others, freeze_token_rep=False):
        """
        Parameters:
        - lr_encoder: Learning rate for the encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: whether the token representation layer should be frozen.
        """
        param_groups = [
            # encoder
            {'params': self.rnn.parameters(), 'lr': lr_others},
            {'params': self.span_rep_layer.parameters(), 'lr': lr_others},
            {'params': self.prompt_rep_layer.parameters(), 'lr': lr_others},
            {"params": self._rel_filtering.parameters(), "lr": lr_others},
            {'params': self.scorer.parameters(), 'lr': lr_others}
        ]

        if not freeze_token_rep:
            # If token_rep_layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({'params': self.token_rep_layer.parameters(), 'lr': lr_encoder})
        else:
            # If token_rep_layer should be frozen, explicitly set requires_grad to False for its parameters
            for param in self.token_rep_layer.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups)

        return optimizer

    def compute_score(self, x):

        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1).to(self.device)  # ([B, num_possible_spans, 2])  *  ([B, num_possible_spans, 1])

        new_length = x['seq_length'].clone()
        new_tokens = []
        all_len_prompt = []
        num_classes_all = []

        # add prompt to the tokens
        for i in range(len(x['tokens'])):
            all_types_i = list(x['classes_to_id'][i].keys())
            # multiple entity types in all_types. Prompt is appended at the start of tokens
            entity_prompt = []
            num_classes_all.append(len(all_types_i))
            # add enity types to prompt
            for relation_type in all_types_i:
                entity_prompt.append(self.rel_token)     # [REL] token
                entity_prompt.append(relation_type)        # relation type
            entity_prompt.append(self.sep_token)         # [SEP] token

            # prompt format:
            # [REL] relation_type [REL] relation_type ... [REL] relation_type [SEP]

            # add prompt to the tokens
            tokens_p = entity_prompt + x['tokens'][i]

            # input format:
            # [REL] relation_type_1 [REL] relation_type_2 ... [REL] relation_type_m [SEP] token_1 token_2 ... token_n

            # update length of the sequence (add prompt length to the original length)
            new_length[i] = new_length[i] + len(entity_prompt)
            # update tokens
            new_tokens.append(tokens_p)
            # store prompt length
            all_len_prompt.append(len(entity_prompt))

        # create a mask using num_classes_all (False, if it exceeds the number of classes, True otherwise)
        max_num_classes = max(num_classes_all)
        rel_type_mask = torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes_all), -1).to(
            x['span_mask'].device)
        rel_type_mask = rel_type_mask < torch.tensor(num_classes_all).unsqueeze(-1).to(
            x['span_mask'].device)  # [batch_size, max_num_classes]

        # compute all token representations
        bert_output = self.token_rep_layer(new_tokens, new_length)
        word_rep_w_prompt = bert_output["embeddings"]  # embeddings for all tokens (with prompt)
        mask_w_prompt = bert_output["mask"]  # mask for all tokens (with prompt)

        # get word representation (after [SEP]), mask (after [SEP]) and relation type representation (before [SEP])
        word_rep = []      # word representation (after [SEP])
        mask = []          # mask (after [SEP])
        rel_type_rep = []  # relation type representation (before [SEP])
        for i in range(len(x['tokens'])):
            prompt_entity_length = all_len_prompt[i]  # length of prompt for this example
            # get word representation (after [SEP])
            word_rep.append(word_rep_w_prompt[i, prompt_entity_length:prompt_entity_length + x['seq_length'][i]])
            # get mask (after [SEP])
            mask.append(mask_w_prompt[i, prompt_entity_length:prompt_entity_length + x['seq_length'][i]])

            # get entity type representation (before [SEP])
            relation_rep = word_rep_w_prompt[i, :prompt_entity_length - 1]  # remove [SEP]
            if self.config.label_embed_strategy == 'ent_token':
                # Only use class token ([REL]) embeddings (every second element starting from 0)
                relation_rep = relation_rep[0::2]
            elif self.config.label_embed_strategy == 'label':
                # Only use label embeddings (every second element starting from 1)
                relation_rep = relation_rep[1::2]
            elif self.config.label_embed_strategy == 'both':
                # Interleave [REL] tokens and label embeddings
                relation_rep = relation_rep.reshape(-1, 2, relation_rep.shape[-1]).mean(dim=1)  # Take the mean of each [REL] and label embedding pair
            rel_type_rep.append(relation_rep)

        # padding for word_rep, mask and rel_type_rep
        word_rep = pad_sequence(word_rep, batch_first=True)          # [B, seq_len, D]
        mask = pad_sequence(mask, batch_first=True)                  # [B, seq_len]
        rel_type_rep = pad_sequence(rel_type_rep, batch_first=True)  # [B, len_types, D]

        # compute span representation
        word_rep = self.rnn(word_rep, mask)                  # ([B, seq_length, D])
        rel_rep = self.span_rep_layer(word_rep, span_idx=span_idx, relations_idx=x['relations_idx'])    # ([B, num_pairs, D])

        # compute final entity type representation (FFN)
        rel_type_rep = self.prompt_rep_layer(rel_type_rep)   # (B, len_types, D)
        num_classes = rel_type_rep.shape[1]                  # number of relation types


        # refine relation representation ##############################################
        relation_classes = x['rel_label']  # [B, num_entity_pairs]
        rel_rep_mask = relation_classes > 0
        ################################################################################

        if hasattr(self, "refine_relation"):
            # refine relation representation
            rel_rep = self.refine_relation(
                rel_rep, word_rep, rel_rep_mask, mask
            )

        if hasattr(self, "refine_prompt"):
            # refine relation representation with relation type representation ############
            rel_type_rep = self.refine_prompt(
                rel_type_rep, rel_rep, rel_type_mask, rel_rep_mask
            )
        ################################################################################


        # Coreference Resolution ##############################
        if hasattr(self, "coref_classifier"):
            coref_scores = self.coref_classifier(rel_rep)  # (B, num_pairs, 1)
        else:
            coref_scores = None
        #######################################################
        
        # similarity score
        scores = self.scorer(rel_rep, rel_type_rep) # ([B, num_pairs, num_classes])

        return scores, num_classes, rel_type_mask, coref_scores  # ([B, num_pairs, num_classes]), num_classes, ([B, num_classes]), ([B, num_pairs, 1])


    def compute_coref_loss(self, coref_scores, coref_ground_truth):

        coref_loss = F.binary_cross_entropy_with_logits(coref_scores, coref_ground_truth)

        return coref_loss

    def compute_relation_loss(self, logits_label, labels_one_hot):

        if self.config.loss_func == "binary_cross_entropy_loss":
            # compute loss (without reduction)
            all_losses = F.binary_cross_entropy_with_logits(
                logits_label, 
                labels_one_hot,
                reduction='none'
            )
        elif self.config.loss_func == "focal_loss":
            # might make it better at long-tail, but overall metrics may become worse
            all_losses = loss_functions.focal_loss_with_logits(
                logits_label, 
                labels_one_hot,
                alpha=self.config.alpha,
                gamma=self.config.gamma,
                reduction='none'
            )
        else:
            raise ValueError(f"Invalid loss function: {self.config.loss_func}")
        
        return all_losses


    def forward(self, x):
        # compute span representation
        scores, num_classes, rel_type_mask, coref_scores = self.compute_score(x)
        batch_size = scores.size(0)


        # loss for filtering classifier
        logits_label = scores.view(-1, num_classes)

        labels = x['rel_label'].view(-1)     # [B * num_entity_pairs]

        # mask for coreference and relation labels
        coref_mask = (labels <= -50) 
        labels[coref_mask] = (labels[coref_mask] // -50).long()                
        rel_mask = (labels != -1)   # Exclude padding (and coreference) labels

        # separate labels for relation classification
        rel_labels = labels.masked_fill(~rel_mask, 0)  # Set non-relation labels to 0

        # one-hot encoding
        labels_one_hot = torch.zeros(labels.size(0), num_classes + 1, dtype=torch.float32).to(scores.device) # ([batch_size * num_spans, num_classes + 1])
        labels_one_hot.scatter_(1, rel_labels.unsqueeze(1), 1) # Set the corresponding index to 1
        labels_one_hot = labels_one_hot[:, 1:]                 # Remove the first column
        # Shape of labels_one_hot: (batch_size * num_spans, num_classes)

        all_losses = self.compute_relation_loss(logits_label, labels_one_hot)
        
        # Mask and weight relation classification loss
        # mask loss using rel_type_mask (B, C)
        masked_loss = all_losses.view(batch_size, -1, num_classes) * rel_type_mask.unsqueeze(1)   #  ([B, L*K, num_classes])  *  ([B, 1, num_classes])
        all_losses = masked_loss.view(-1, num_classes)
        # expand mask_label to all_losses
        rel_mask = rel_mask.unsqueeze(-1).expand_as(all_losses)
        
        # Assign weights based on the true labels
        weight_c = labels_one_hot * self.positive_weight + (1 - labels_one_hot) * self.negative_weight

        # apply mask
        all_losses = all_losses * rel_mask.float() * weight_c
        rel_loss = all_losses.sum()
        total_loss = rel_loss

        if hasattr(self, "coref_classifier"):
            if coref_mask.sum() == 0:
                logger.info("Coreference mask is empty, skipping coreference loss calculation")
            else:
                # compute coreference loss (masked for coreference labels only)
                coref_scores = coref_scores.view(-1)                                # Flatten coref_scores to match label shape (B * num_pairs * 1)

                # Prepare coref_ground_truth with valid target values (i.e, set -2 -> 1 and others -> 0)
                coref_ground_truth = torch.zeros_like(labels, dtype=torch.float32)
                coref_ground_truth[coref_mask] = 1.0

                valid_mask = (labels != -1)

                coref_loss = self.compute_coref_loss(coref_scores[valid_mask], coref_ground_truth[valid_mask])
                coref_loss = self.config.coref_loss_weight * coref_loss
                total_loss += coref_loss
                return {'total_loss': total_loss, 'coref_loss': coref_loss, 'rel_loss': rel_loss}

        return {'total_loss': total_loss} # total_loss is rel_loss if no coref_classifier
    

    @torch.no_grad()
    def predict(self, x, flat_ner=False, threshold: list | float = 0.5, ner=None):
        self.eval()
        
        # start_time = time.time()
        
        local_scores, num_classes, rel_type_mask, coref_scores = self.compute_score(x)
        
        # forward_pass_time = time.time() - start_time
        # logger.info(f"[Optimized Predict] Time taken for forward pass: {forward_pass_time:.4f} seconds")

        if isinstance(threshold, float):
            threshold = [threshold]
        
        probabilities = torch.sigmoid(local_scores)  # Shape: [batch_size, num_pairs, num_classes]
        
        # Start timing thresholding without synchronization
        # start = time.time()
        
        # Build a global mapping from class names to IDs
        all_class_names = set()
        for classes in x['classes_to_id']:
            all_class_names.update(classes.keys())
        class_name_to_id = {name: idx for idx, name in enumerate(sorted(all_class_names))}
        id_to_class_name = {idx: name for name, idx in class_name_to_id.items()}
        num_classes_total = len(class_name_to_id)

        # Convert types_list to tensors of class IDs
        types_list = [torch.tensor([class_name_to_id[name] for name in classes.keys()], device=probabilities.device) for classes in x['classes_to_id']]
        max_num_classes = max(len(types) for types in types_list)
        padded_types_tensor = torch.full((len(types_list), max_num_classes), -1, dtype=torch.long, device=probabilities.device)
        for idx, types in enumerate(types_list):
            padded_types_tensor[idx, :types.size(0)] = types

        # Convert x['relations_idx'] to a padded tensor
        relations_idx_list = x['relations_idx']
        max_num_pairs = probabilities.shape[1]  # Should match num_pairs in local_scores
        # Determine the maximum shape beyond the first dimension
        max_relations_idx_shape = relations_idx_list[0].shape[1:]  # Exclude the first dimension (num_pairs)

        # Initialize the padded tensor with the maximum shape
        padded_shape = (len(relations_idx_list), max_num_pairs) + max_relations_idx_shape
        padded_relations_idx = torch.full(padded_shape, -1, dtype=torch.long, device=probabilities.device)

        # Pad and assign each relations_idx tensor
        for idx, relations_idx in enumerate(relations_idx_list):
            num_pairs = relations_idx.shape[0]
            padded_relations_idx[idx, :num_pairs, ...] = relations_idx.to(probabilities.device)

        # logger.info(f"## Max Number of entity pairs: {max_num_pairs}")
        # logger.info(f"## Max Number of classes: {max_num_classes}")

        # Initialize the dictionary to store results per threshold
        rels_per_threshold = {}
        for thresh in threshold:
            # Apply threshold
            triggered_relations = probabilities > thresh  # Shape: [batch_size, num_pairs, num_classes]

            # Build a mask for valid class indices
            batch_sizes = torch.tensor([len(types) for types in types_list], device=probabilities.device)
            class_mask = torch.arange(max_num_classes, device=probabilities.device).unsqueeze(0).unsqueeze(0) < batch_sizes.unsqueeze(1).unsqueeze(2)
            triggered_relations &= class_mask

            # Get indices where relations are triggered
            batch_indices, pair_indices, class_indices = torch.nonzero(triggered_relations, as_tuple=True)

            if batch_indices.numel() == 0:
                rels = [[] for _ in range(len(x["tokens"]))]
                rels_per_threshold[thresh] = rels
                continue

            # Get scores
            scores = probabilities[batch_indices, pair_indices, class_indices]
            
            # Map class indices to actual class IDs
            class_ids = padded_types_tensor[batch_indices, class_indices]
            
            # Remove invalid class IDs (i.e., where class_id == -1)
            valid_class_mask = class_ids != -1
            batch_indices = batch_indices[valid_class_mask]
            pair_indices = pair_indices[valid_class_mask]
            class_ids = class_ids[valid_class_mask]
            scores = scores[valid_class_mask]

            if batch_indices.numel() == 0:
                rels = [[] for _ in range(len(x["tokens"]))]
                rels_per_threshold[thresh] = rels
                continue

            # Get corresponding entity pairs
            entity_pairs = padded_relations_idx[batch_indices, pair_indices]  # Shape: [num_relations, ...]

            # Build a mask for valid entity pairs (entity_pair does not contain -1)
            valid_entity_mask = (entity_pairs != -1).all(dim=tuple(range(1, entity_pairs.dim())))
            batch_indices = batch_indices[valid_entity_mask]
            pair_indices = pair_indices[valid_entity_mask]
            entity_pairs = entity_pairs[valid_entity_mask]
            class_ids = class_ids[valid_entity_mask]
            scores = scores[valid_entity_mask]

            if batch_indices.numel() == 0:
                rels = [[] for _ in range(len(x["tokens"]))]
                rels_per_threshold[thresh] = rels
                continue

            # Convert class IDs to class names
            class_ids_cpu = class_ids.cpu()
            relation_types = [id_to_class_name[class_id.item()] for class_id in class_ids_cpu]

            # Collect relations per example
            rels = [[] for _ in range(len(x["tokens"]))]
            batch_indices_cpu = batch_indices.cpu()
            entity_pairs_cpu = entity_pairs.cpu()
            scores_cpu = scores.cpu()
            for idx in range(batch_indices.size(0)):
                i = batch_indices_cpu[idx].item()
                entity_pair = entity_pairs_cpu[idx].tolist()
                relation_type = relation_types[idx]
                score = scores_cpu[idx].item()
                rels[i].append((entity_pair, relation_type, score))

            rels_per_threshold[thresh] = rels


        # thresholding_time = time.time() - start
        # logger.info(f"[Optimized Predict] Time taken for thresholding: {thresholding_time:.4f} seconds")
        
        return rels_per_threshold if len(threshold) > 1 else rels_per_threshold[threshold[0]]



    def predict_relations(self, text, labels, flat_ner=True, threshold=0.5, ner=None, top_k=-1):
        return self.batch_predict_relations([text], labels, flat_ner=flat_ner, threshold=threshold, ner=[ner], top_k=top_k)[0]

    def batch_predict_relations(self, texts, labels, flat_ner=True, threshold=0.5, ner=None, top_k=-1):
        """
        Predict relations for a batch of texts.
        texts:  List of texts | List[str]
        labels: List of labels | List[str]
        ...
        """

        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            if type(text) is str:
                for match in re.finditer(r'\w+(?:[-_]\w+)*|\S', text):
                    tokens.append(match.group())
                    start_token_idx_to_text_idx.append(match.start())
                    end_token_idx_to_text_idx.append(match.end())
            else:
                tokens = text  # already tokenized
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        if ner is not None:
            for i, x in enumerate(input_x):
                x['ner'] = ner[i]

        x = self.collate_fn(input_x, labels, device=self.device)
        assert all([l in labels for l in x['classes_to_id'][0].keys()]), f"Labels mismatch after collate function: {labels} != {x['classes_to_id'][0].keys()}. `config.fixed_relation_types` is likely wrong"
        outputs = self.predict(x, flat_ner=flat_ner, threshold=threshold, ner=ner)

        # retrieve top_k predictions (if top_k > -1)
        if top_k > 0:
            top_k_outputs = []
            for i, output in enumerate(outputs):

                # sort output by score
                output = sorted(output, key=lambda x: x[2], reverse=True)

                rels = []
                position_set = {}  # track all position predictions to take top_k predictions
                for rel in output:
                    (head_pos, tail_pos), pred_label, score = rel

                    hashable_positions = (tuple(head_pos), tuple(tail_pos))
                    if hashable_positions not in position_set:
                        position_set[hashable_positions] = 0

                    if position_set[hashable_positions] < top_k:
                        rels.append(rel)
                        position_set[hashable_positions] += 1

                top_k_outputs.append(rels)
            outputs = top_k_outputs

        all_relations = []

        for i, output in enumerate(outputs):

            rels = []
            for (head_pos, tail_pos), pred_label, score in output:

                # +1 to indices to restore spaCy tokenization
                rel = {
                    'head_pos' : [head_pos[0], head_pos[1]+1],
                    'tail_pos' : [tail_pos[0], tail_pos[1]+1],
                    'head_text' : texts[i][head_pos[0]:head_pos[1]+1],
                    'tail_text' : texts[i][tail_pos[0]:tail_pos[1]+1],
                    'label': pred_label,
                    'score': score,
                }
                
                rels.append(rel)

            all_relations.append(rels)

        return all_relations


    def evaluate(
            self, test_data, flat_ner=False, 
            threshold: list | float = 0.5, batch_size=12, relation_types=None, 
            top_k=1, return_preds=False, dataset_name: str = None, num_eval_samples: int = None
        ):
        # NOTE: observing that subsequent evaluations are faster than the first one and not sure why
        self.eval()
        if len(relation_types) > 0:
            logger.info(f"Number of classes to evaluate with --> {len(relation_types)}")
        else:
            logger.info(f"No fixed relation types given for evaluation. Sampling relation types from the batch.")
        data_loader = self.create_dataloader(test_data, batch_size=batch_size, relation_types=relation_types, shuffle=False)

        threshold = [threshold] if isinstance(threshold, float) else threshold
        all_preds = {thresh: [] for thresh in threshold}
        all_trues = {thresh: [] for thresh in threshold}

        # if the dataset is streamable, num_eval_samples should be provided
        len_data_loader = num_eval_samples or len(data_loader)
        with tqdm(total=len_data_loader, desc="Evaluating") as pbar:
            for i, x in enumerate(data_loader):
                if i == 0:
                    classes = list(x['classes_to_id'][0].keys())
                    logger.info(f"## Evaluation x['classes_to_id'][0] (showing {min(15, len(classes))}/{len(classes)}) --> {classes[:min(15, len(classes))]}")
                ner = x['entities']

                batch_predictions = self.predict(x, flat_ner, threshold, ner)
                if isinstance(batch_predictions, list):
                    # only one threshold was used
                    batch_predictions = {threshold[0]: batch_predictions}

                
                # for all thresholds, collect predictions and true relations
                for thresh, batch_preds in batch_predictions.items():
                    all_trues[thresh].extend(x["relations"])
                    # format relation predictions for metrics calculation
                    batch_predictions_formatted = []
                    for i, output in enumerate(batch_preds):

                        # sort output by score
                        output = sorted(output, key=lambda x: x[2], reverse=True)

                        rels = []
                        position_set = {}  # track all position predictions to take top_k predictions
                        for (head_pos, tail_pos), pred_label, score in output:

                            hashable_positions = (tuple(head_pos), tuple(tail_pos))
                            if hashable_positions not in position_set:
                                position_set[hashable_positions] = 0

                            if position_set[hashable_positions] < top_k:

                                rel = {
                                    'head' : {'position': head_pos},
                                    'tail' : {'position': tail_pos},
                                    'relation_text': pred_label,
                                    'score': score,
                                }
                                
                                rels.append(rel)
                                position_set[hashable_positions] += 1

                        batch_predictions_formatted.append(rels)

                    all_preds[thresh].extend(batch_predictions_formatted)
                    
                pbar.update(1)
                

        best_threshold = None
        best_out, best_metric_dict = None, None
        for thresh, preds in all_preds.items():
            evaluator = RelEvaluator(all_trues[thresh], preds, dataset_name=dataset_name)
            out, metric_dict = evaluator.evaluate()
            if best_out is None or metric_dict[self.threshold_search_metric] > best_metric_dict[self.threshold_search_metric]:
                best_out = out
                best_metric_dict = metric_dict
                best_threshold = thresh
        best_metric_dict["best_threshold"] = best_threshold
        
        if return_preds:
            return best_out, best_metric_dict, all_preds
        return best_out, best_metric_dict


    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        if 'glirel_beta' in model_id:
            assert int(glirel.__version__.split('.')[0]) == 0, f"glirel_beta is only compatible with glirel<1.0.0. Detected version: {glirel.__version__}"

        # Use "pytorch_model.bin" and "glirel_config.json"
        model_file = Path(model_id) / "pytorch_model.bin"
        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config_file = Path(model_id) / "glirel_config.json"
        if not config_file.exists():
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="glirel_config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config = load_config_as_namespace(config_file)
        model = cls(config)
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict, assign=True)
        model.to(map_location)
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            if isinstance(config, argparse.Namespace):
                config = vars(config)
            (save_directory / "glirel_config.json").write_text(json.dumps(config, indent=2))

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def to(self, device):
        super().to(device)
        import flair

        flair.device = device
        self.device = device
        return self


def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
