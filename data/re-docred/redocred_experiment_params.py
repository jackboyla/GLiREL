

REDOCRED_EXP_SWEEP_CONFIG = {
    "metric": {"goal": "maximize", "name": "eval_f1_macro"},
    "parameters": {
        'seed': {"values": [1, 5, 619, 999, 111]},
        # "refine_prompt": {"values": [False, True]},
        # "refine_relation": {"values": [False, True]},
        # "span_marker_mode": {"values": ["markerv1", "markerv2"]},
        # "add_entity_markers": {"values": [False, True]},
        # "label_embed_strategy": {"values": ["label"]},
        # "num_unseen_rel_types": {"values": [15, 10, 5]},
        "prev_path": {"values": ["logs/zero_rel/zero_rel-2024-10-06__21-28-09/saved_at/model_70000", "none"]},
    },
}