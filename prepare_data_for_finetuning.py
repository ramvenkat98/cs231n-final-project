import json
from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVista")

L = []
for d in dataset["testmini"]:
    if d['question_type'] != 'multi_choice':
      continue
    id = d['pid']
    image = '/home/ramvenkat98/cs231n-final-project/cs231n-final-project/image_data/' + d['image']
    q = d['query']
    text = '<ImageHere>' + q
    correct_choice =  chr(ord('A') + d['choices'].index(d['answer']))
    a = f'The answer is ({correct_choice}).'
    L += [
      {
        "id": id,
        "image": [image],
        "conversations": [
          {
            "from": "user",
            "value": text,
          },
          {
            "from": "assistant",
            "value": a
          },
        ]
      },
    ]

with open('testmini_example.json', 'w') as file:
    json.dump(L, file, indent = 4)
