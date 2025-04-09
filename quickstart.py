from glirel import GLiREL

model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")

tokens = ["The", "race", "took", "place", "between", "Godstow", "and", "Binsey", "along", "the", "Upper", "River", "Thames", "."]

labels = ['country of origin', 'located in or next to body of water', 'licensed to broadcast to', 'father', 'followed by', 'characters', ]

ner = [
    [7, 7, "Q4914513", "Binsey"], 
    [11, 12, "Q19686", "River Thames"]
]

ground_truth_relations = [
    {
      "head": {"mention": "Binsey", "position": [7, 7], "type": "LOC"}, # 'type' is not used -- it can be any string!
      "tail": {"mention": "River Thames", "position": [11, 12], "type": "Q19686"}, 
      "relation_text": "located in or next to body of water"
    }
]

relations, loss = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=-1, ground_truth_relations=ground_truth_relations)
# relations = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=-1)

print('Number of relations:', len(relations))

sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
print("\nDescending Order by Score:")
for item in sorted_data_desc:
    print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | score: {item['score']}")

print("Success! ✅")