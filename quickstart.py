from glirel import GLiREL
import spacy

model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
nlp = spacy.load('en_core_web_sm')

text = 'Derren Nesbitt had a history of being cast in "Doctor Who", having played villainous warlord Tegana in the 1964 First Doctor serial "Marco Polo".'
doc = nlp(text)
tokens = [token.text for token in doc]

labels = ['country of origin', 'licensed to broadcast to', 'father', 'followed by', 'characters']

ner = [[26, 27, 'PERSON', 'Marco Polo'], [22, 23, 'Q2989412', 'First Doctor']] # 'type' is not used -- it can be any string!

relations = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=1)

print('Number of relations:', len(relations))

sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
print("\nDescending Order by Score:")
for item in sorted_data_desc:
    print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | score: {item['score']}")

print("Success! ✅")