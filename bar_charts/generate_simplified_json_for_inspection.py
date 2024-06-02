import json

with open('bar_charts_visual_linguistic_train.json', 'r') as f:
    L = json.load(f)

simplified = []

for x in L:
    simplified.append(
        {
            "id": x['id'],
            "question": x['question'],
            "answer": x['answer'],
            "choices": x['choices'],
        }
    )

with open('simplified_bar_charts.json', 'w') as f:
    json.dump(simplified, f, indent = 4)
