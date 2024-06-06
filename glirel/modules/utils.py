import numpy as np
import torch
import torch.nn.functional as F


def down_weight_loss(logits, y, sample_rate=0.1, is_logit=True):
    rate = 1 - sample_rate

    if is_logit:
        loss_func = F.cross_entropy
    else:
        loss_func = F.nll_loss

    loss_entity = loss_func(logits, y.masked_fill(y == 0, -1), ignore_index=-1, reduction='sum')
    loss_non_entity = loss_func(logits, y.masked_fill(y > 0, -1), ignore_index=-1, reduction='sum')

    return loss_entity + loss_non_entity * rate


def get_relations(x, rel_logits, topk_pair_idx, max_top_k, candidate_spans_idx, threshold=0.5, output_confidence=False):
    rel_prob = torch.sigmoid(rel_logits)  # B, N, C

    relations = [[] for i in range(len(rel_logits))]

    rel_no = (rel_prob > threshold).nonzero(as_tuple=True)

    for bt, pos, cls in zip(*rel_no):
        lb = x["id_to_classes"][cls.item() + 1]
        pair_idx_pred = topk_pair_idx[bt, pos].item()
        head, tail = np.unravel_index(pair_idx_pred, (max_top_k, max_top_k))
        head = tuple(candidate_spans_idx[bt, head].tolist())
        tail = tuple(candidate_spans_idx[bt, tail].tolist())

        confidence = rel_prob[bt, pos, cls].item()

        if output_confidence:
            relations[bt.item()].append((head, tail, lb, confidence))
        else:
            relations[bt.item()].append((head, tail, lb))

    return relations


def get_relation_with_span(x):
    entities, relations = x['entities'], x['relations']
    B = len(entities)
    relation_with_span = [[] for i in range(B)]
    for i in range(B):
        rel_i = relations[i]
        ent_i = entities[i]
        for rel in rel_i:
            act = (ent_i[rel[0]], ent_i[rel[1]], rel[2])
            relation_with_span[i].append(act)
    return relation_with_span


def get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label):
    B, max_top_k = candidate_span_label.shape

    relation_classes = torch.zeros((B, max_top_k, max_top_k), dtype=torch.long, device=candidate_spans_idx.device)

    # Populate relation classes
    for i in range(B):
        rel_i = x["relations"][i]
        ent_i = x["entities"][i]

        new_heads, new_tails, new_rel_type = [], [], []

        # Loop over the relations and entities to populate initial lists
        for k in rel_i:
            heads_i = [ent_i[k[0]][0], ent_i[k[0]][1]]
            tails_i = [ent_i[k[1]][0], ent_i[k[1]][1]]
            type_i = k[2]
            new_heads.append(heads_i)
            new_tails.append(tails_i)
            new_rel_type.append(type_i)

        # Update the original lists
        heads_, tails_, rel_type = new_heads, new_tails, new_rel_type

        # idx of candidate spans
        cand_i = candidate_spans_idx[i].tolist()

        for heads_i, tails_i, type_i in zip(heads_, tails_, rel_type):

            flag = False
            if isinstance(x["classes_to_id"], dict):
                if type_i in x["classes_to_id"]:
                    flag = True
            elif isinstance(x["classes_to_id"], list):
                if type_i in x["classes_to_id"][i]:
                    flag = True

            if heads_i in cand_i and tails_i in cand_i and flag:
                idx_head = cand_i.index(heads_i)
                idx_tail = cand_i.index(tails_i)

                if isinstance(x["classes_to_id"], list):
                    relation_classes[i, idx_head, idx_tail] = x["classes_to_id"][i][type_i]
                elif isinstance(x["classes_to_id"], dict):
                    relation_classes[i, idx_head, idx_tail] = x["classes_to_id"][type_i]

    # flat relation classes
    relation_classes = relation_classes.view(-1, max_top_k * max_top_k)

    # put to -1 class where corresponding candidate_span_label is -1 (for both head and tail)
    head_candidate_span_label = candidate_span_label.view(B, max_top_k, 1).repeat(1, 1, max_top_k).view(B, -1)
    tail_candidate_span_label = candidate_span_label.view(B, 1, max_top_k).repeat(1, max_top_k, 1).view(B, -1)

    relation_classes.masked_fill_(head_candidate_span_label.view(B, max_top_k * max_top_k) == -1, -1)  # head
    relation_classes.masked_fill_(tail_candidate_span_label.view(B, max_top_k * max_top_k) == -1, -1)  # tail

    return relation_classes


def _get_candidates(sorted_idx, tensor_elem, topk=10):
    # sorted_idx [B, num_spans]
    # tensor_elem [B, num_spans, D] or [B, num_spans]

    sorted_topk_idx = sorted_idx[:, :topk]

    if len(tensor_elem.shape) == 3:
        B, num_spans, D = tensor_elem.shape
        # [B, topk, D]
        topk_tensor_elem = tensor_elem.gather(1, sorted_topk_idx.unsqueeze(-1).expand(-1, -1, D))
    else:
        B, num_spans = tensor_elem.shape
        # [B, topk]
        topk_tensor_elem = tensor_elem.gather(1, sorted_topk_idx)

    return topk_tensor_elem, sorted_topk_idx