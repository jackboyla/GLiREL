from collections import defaultdict

import numpy as np
import os
import json
import torch
from seqeval.metrics.v1 import _prf_divide


class RelEvaluator:
    def __init__(self, all_true, all_outs, dataset_name: str = None):
        self.all_true = all_true
        self.all_outs = all_outs
        self.dataset_name = dataset_name

    def get_relations_fr(self, rels):
        all_rels = []
        for r in rels:
            all_rels.append(
                [
                    r['relation_text'], 
                    tuple(r['head']['position']), 
                    tuple(r['tail']['position'])
                ]
            )
        return all_rels

    def transform_data(self):
        all_true_rel = []
        all_outs_rel = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_relations_fr(i)
            all_true_rel.append(e)
            e = self.get_relations_fr(j)
            all_outs_rel.append(e)
        
        # # DEBUG: find all the relations we missed or got wrong
        # for i, (true, pred) in enumerate(zip(all_true_rel, all_outs_rel)):
        #     instance_all_true_set = set([tuple(t) for t in true])
        #     instance_out_set = set([tuple(t) for t in pred])
        #     assert len(instance_all_true_set) == len(true), f"Duplicate relations in true data for instance {i}"
        #     assert len(instance_out_set) == len(pred), f"Duplicate relations in predicted data for instance {i}"
        #     fn, tp, fp = [], [], []
        #     for p in instance_out_set:
        #         if p in instance_all_true_set:
        #             tp.append(p)
        #         else:
        #             fp.append(p)
        #     for t in instance_all_true_set:
        #         if t not in instance_out_set:
        #             fn.append(t)
        #     import ipdb; ipdb.set_trace()
                
        return all_true_rel, all_outs_rel

    @torch.no_grad()
    def evaluate(self):
        all_true_typed, all_outs_typed = self.transform_data()
        micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = self.compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"Micro P: {micro_precision:.2%}\tMicro R: {micro_recall:.2%}\tMicro F1: {micro_f1:.2%}\n"
        output_str += f"Macro P: {macro_precision:.2%}\tMacro R: {macro_recall:.2%}\tMacro F1: {macro_f1:.2%}\n"
        metric_dict = {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }
        return output_str, metric_dict
    
    def extract_tp_actual_correct(self, y_true, y_pred):
        # y_pred[0] -> ['work location', (19, 20), (23, 24), 0]
        relations_true = defaultdict(set)
        relations_pred = defaultdict(set)

        for type_name, head, tail, idx in y_true:
            relations_true[type_name].add((head, tail, idx))
        for type_name, head, tail, idx in y_pred:
            # NOTE: we are only interested in the evaluating against 
            # annotated relations that are present in the true data (i.e. not the ones that are not annotated in the case of FewRel)
            if self.dataset_name in ["few_rel", "wiki_zsl", "redocred"]:
                if any((head, tail, idx) in relations_true[t] for t in relations_true.keys()):
                    relations_pred[type_name].add((head, tail, idx))
            else:
                relations_pred[type_name].add((head, tail, idx))


        target_names = sorted(set(relations_true.keys()) | set(relations_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            relations_true_type = relations_true.get(type_name, set())
            relations_pred_type = relations_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(relations_true_type & relations_pred_type))
            pred_sum = np.append(pred_sum, len(relations_pred_type))
            true_sum = np.append(true_sum, len(relations_true_type))

        return pred_sum, tp_sum, true_sum, target_names

    def flatten_for_eval(self, y_true, y_pred):
        all_true = []
        all_pred = []

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            all_true.extend([t + [i] for t in true])
            all_pred.extend([p + [i] for p in pred])

        return all_true, all_pred


    def compute_prf(self, y_true, y_pred):
        # macro will weight all classes equally, micro will weight all instances equally (regardless of class)
        # ref (because I always forget) -> https://datascience.stackexchange.com/a/24051
        y_true, y_pred = self.flatten_for_eval(y_true, y_pred)

        pred_sum, tp_sum, true_sum, target_names = self.extract_tp_actual_correct(y_true, y_pred)

            
        # Macro averaging calculates the metrics for each class separately and then average them
        macro_f_score, macro_recall, macro_precision = [], [], []
        for i in range(len(tp_sum)):
            p = _prf_divide(numerator=np.array([tp_sum[i]]), denominator=np.array([pred_sum[i]]), metric='precision', modifier='predicted', average='macro', warn_for=('precision',), zero_division='warn')
            r = _prf_divide(numerator=np.array([tp_sum[i]]), denominator=np.array([true_sum[i]]), metric='recall', modifier='true', average='macro', warn_for=('recall',), zero_division='warn')
            f = 2 * (p * r) / (p + r) if p + r != 0 else np.array([0])
            macro_precision.append(p)
            macro_recall.append(r)
            macro_f_score.append(f)
        macro_precision = [np.mean(macro_precision)]
        macro_recall = [np.mean(macro_recall)]
        macro_f_score = [np.mean(macro_f_score)]


        # Micro averaging is simply the total number of true positives, false positives, and false negatives
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

        micro_precision = _prf_divide(
            numerator=tp_sum,
            denominator=pred_sum,
            metric='precision',
            modifier='predicted',
            average='micro',
            warn_for=('precision', 'recall', 'f-score'),
            zero_division='warn'
        )

        micro_recall = _prf_divide(
            numerator=tp_sum,
            denominator=true_sum,
            metric='recall',
            modifier='true',
            average='micro',
            warn_for=('precision', 'recall', 'f-score'),
            zero_division='warn'
        )

        denominator = micro_precision + micro_recall
        denominator[denominator == 0.] = 1
        micro_f_score = 2 * (micro_precision * micro_recall) / denominator


        return {'micro_precision': micro_precision[0], 'micro_recall': micro_recall[0], 'micro_f_score': micro_f_score[0],
                'macro_precision': macro_precision[0], 'macro_recall': macro_recall[0], 'macro_f_score': macro_f_score[0],
                }



def is_nested(idx1, idx2):
    # Return True if idx2 is nested inside idx1 or vice versa
    return (idx1[0] <= idx2[0] and idx1[1] >= idx2[1]) or (idx2[0] <= idx1[0] and idx2[1] >= idx1[1])


def has_overlapping(idx1, idx2):
    overlapping = True
    if idx1[:2] == idx2[:2]:
        return overlapping
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]):
        overlapping = False
    return overlapping


def has_overlapping_nested(idx1, idx2):
    # Return True if idx1 and idx2 overlap, but neither is nested inside the other
    if idx1[:2] == idx2[:2]:
        return True
    if ((idx1[0] > idx2[1] or idx2[0] > idx1[1]) or is_nested(idx1, idx2)) and idx1 != idx2:
        return False
    else:
        return True


def greedy_search(spans, flat_ner=True):  # start, end, class, score

    if flat_ner:
        has_ov = has_overlapping
    else:
        has_ov = has_overlapping_nested

    new_list = []
    span_prob = sorted(spans, key=lambda x: -x[-1])
    for i in range(len(spans)):
        b = span_prob[i]
        flag = False
        for new in new_list:
            if has_ov(b[:-1], new):
                flag = True
                break
        if not flag:
            new_list.append(b[:-1])
    new_list = sorted(new_list, key=lambda x: x[0])
    return new_list
