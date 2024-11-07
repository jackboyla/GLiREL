import numpy as np
import torch
import torch.nn.functional as F


def remove_duplicates(data):
    """
    - Check for and remove duplicate spans and relations in the data.
    """

    for i, item in enumerate(data):
        # Remove duplicate relations
        relation_pos = set()
        unique_relations = []
        for r in item['relations']:
            position_tuple = (tuple(r['head']['position']), tuple(r['tail']['position']))
            if position_tuple not in relation_pos:
                relation_pos.add(position_tuple)
                unique_relations.append(r)
            else:
                print(f"Duplicate relation removed in (idx {i}) Relation --> {r}")
        item['relations'] = unique_relations  # Update relations with unique list

        # Remove duplicate spans
        span_set = set()
        unique_spans = []
        for span in item['ner']:
            span_pos = (span[0], span[1])
            if span_pos not in span_set:
                span_set.add(span_pos)
                unique_spans.append(span)
            else:
                print(f"Duplicate span removed in (idx {i}) Span --> {span}")
        item['ner'] = unique_spans  # Update NER spans with unique list

    return data


def sanity_check_data(data):
    """
    - Check for duplicate spans and relations in the data.
    - Ensure that the relation positions are found in the NER spans.
    """

    for i, item in enumerate(data):
        relation_pos = set()
        for r in item['relations']:
            position_tuple = (tuple(r['head']['position']), tuple(r['tail']['position']))
            # duplicate relations
            assert position_tuple not in relation_pos, f"Duplicate position for relation in (idx {i}) Relation --> {r}"
            relation_pos.add(position_tuple)

        span_set = set()
        for span in item['ner']:
            span_pos = (span[0], span[1])
            # duplicate spans
            assert span_pos not in span_set, f"Duplicate span in (idx {i}) Span --> {span}"
            span_set.add(span_pos)


        for pos_tuple in relation_pos:
            for pos in pos_tuple:
                # relation position not found in NER spans
                assert pos in span_set, f"Relation position not found in NER spans in (idx {i}) Relation position --> {pos}"


def constrain_relations_by_entity_type(ents, labels, relations):
    '''
    relations: {'head_pos': [15, 17], 'tail_pos': [25, 26], 'head_text': ['April', '1976'], 'tail_text': ['California'], 'label': 'headquartered in', 'score': 0.9820516705513}
    labels: {'father': {'allowed_head': ['PERSON'], 'allowed_tail': ['PERSON']}}
    '''
    ner = {(ent.start, ent.end): ent.label_ for ent in ents}
    rel_types = list(labels.keys())
    constrained_relations = []
    for relation in relations:
        head_label = ner[(relation['head_pos'][0], relation['head_pos'][1])]
        tail_label = ner[(relation['tail_pos'][0], relation['tail_pos'][1])]
        if head_label in labels[relation['label']].get('allowed_head', rel_types) and tail_label in labels[relation['label']].get('allowed_tail', rel_types):
            constrained_relations.append(relation)
    
    return constrained_relations

def get_entity_position(entity):
    return tuple(entity["position"])
    
def get_coreference_clusters(relations_list_of_lists, coreference_label="SELF"):
    """
    Generates coreference clusters from relations based on "SELF" relationships.

    Parameters:
    - relations: List of List of relation dictionaries.

    Returns:
    - sorted_clusters: List of List of clusters, each cluster is a list of entity positions.
    - entity_to_cluster_idx: List of Dictionary mapping entity positions to cluster indices.
    """
    if isinstance(relations_list_of_lists[0], dict):
        relations_list_of_lists = [relations_list_of_lists]

    sorted_clusters_list, entity_to_cluster_idx_list = [], []
    for relations in relations_list_of_lists:
        # Collect all unique entities
        entities = set()
        for relation in relations:
            head_pos = get_entity_position(relation["head"]) if "head" in relation else tuple(relation["head_pos"])
            tail_pos = get_entity_position(relation["tail"]) if "tail" in relation else tuple(relation["tail_pos"])
            entities.add(head_pos)
            entities.add(tail_pos)

        # Initialize Union-Find structure
        parent = {entity: entity for entity in entities}

        def find(u):
            # Path compression
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv

        # Union entities connected by "SELF" relationships
        for relation in relations:
            relation_is_coreference = (relation["relation_text"] == coreference_label) if "relation_text" in relation else (relation['label'] == coreference_label)
            if relation_is_coreference:
                head_pos = get_entity_position(relation["head"]) if "head" in relation else tuple(relation["head_pos"])
                tail_pos = get_entity_position(relation["tail"]) if "tail" in relation else tuple(relation["tail_pos"])
                union(head_pos, tail_pos)

        # Build clusters based on connected components
        clusters = {}
        for entity in entities:
            root = find(entity)
            clusters.setdefault(root, []).append(entity)

        # Sort clusters by the earliest mention position
        sorted_clusters = sorted(
            clusters.values(),
            key=lambda cluster: min(pos[0] for pos in cluster)
        )

        # Create a mapping from entity positions to cluster indices
        entity_to_cluster_idx = {}
        for idx, cluster_entities in enumerate(sorted_clusters):
            for entity in cluster_entities:
                entity_to_cluster_idx[entity] = idx

        sorted_clusters_list.append(sorted_clusters)
        entity_to_cluster_idx_list.append(entity_to_cluster_idx)

    return sorted_clusters_list, entity_to_cluster_idx_list 

def aggregate_cluster_relations(entity_to_cluster_idx_list, relations_list, coreference_label="SELF"):
    """
    Aggregates relations across clusters based on the given clusters and relations.

    Parameters:
    - clusters: List of clusters, each cluster is a list of entity positions.
    - entity_to_cluster_idx: Dictionary mapping entity positions to cluster indices.
    - relations: Original list of relation dictionaries.

    Returns:
    - cluster_relations: List of aggregated cluster-to-cluster relations.
    """
    if isinstance(relations_list[0], dict):
        relations_list = [relations_list]
    if isinstance(entity_to_cluster_idx_list, dict):
        entity_to_cluster_idx_list = [entity_to_cluster_idx_list]

    cluster_relations_list = []
    for entity_to_cluster_idx, relations in zip(entity_to_cluster_idx_list, relations_list):
        # Initialize a set to avoid duplicates
        seen_relations = set()
        cluster_relations = []

        for relation in relations:
            # Skip "SELF" relations as they are used for coreference clustering
            relation_is_coreference = (relation["relation_text"] == coreference_label) if "relation_text" in relation else (relation['label'] == coreference_label)
            if relation_is_coreference:
                continue

            head_pos = get_entity_position(relation["head"]) if "head" in relation else tuple(relation["head_pos"])
            tail_pos = get_entity_position(relation["tail"]) if "tail" in relation else tuple(relation["tail_pos"])
            try:
                h_idx = entity_to_cluster_idx[head_pos]
                t_idx = entity_to_cluster_idx[tail_pos]
            except:
                print("relation", relation)
                print("head_pos", head_pos)
                print("tail_pos", tail_pos)
                print("entity_to_cluster_idx", entity_to_cluster_idx)
                raise
            r_text = relation["relation_text"] if "relation_text" in relation else relation['label']

            # Create the aggregated relation
            aggregated_relation = {
                "h_idx": h_idx,
                "t_idx": t_idx,
                "r": r_text
            }

            # Include additional fields if available
            for field in ["title", "evidence"]:
                if field in relation:
                    aggregated_relation[field] = relation[field]

            # Avoid duplicate relations
            relation_key = (h_idx, t_idx, r_text)
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                cluster_relations.append(aggregated_relation)

        cluster_relations = sorted(cluster_relations, key=lambda x: (x["h_idx"], x["t_idx"], x["r"]))
        cluster_relations_list.append(cluster_relations)
    
    return cluster_relations_list

# from EnriCo --> https://github.com/urchade/EnriCo/blob/main/modules/utils.py

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