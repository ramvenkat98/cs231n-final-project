import json
import ast
import random

randomize_ordering = True

LINE_CHARTS_ORIGINAL_JSON_FILE_PATH = '/home/ramvenkat98/cs231n-final-project/line_charts/line_charts_visual_linguistic_test.json'
LINE_CHARTS_PROCESSED_FILE_PATH = 'line_charts_processed_visual_linguistic_val_randomized.json'
IMAGE_PATH = '/home/ramvenkat98/cs231n-final-project/'
BLOCKLIST = [8, 42, 46, 52, 3]

with open(LINE_CHARTS_ORIGINAL_JSON_FILE_PATH, 'r') as file:
    data = json.load(file)

STATS_BY_NUM_CHOICES = {}

def generate_formatted_question_and_answer(image, question, choices, answer, id):
    hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    question_formatted = f"Question: {question}"
    options_formatted = "Choices:"
    for (i, choice) in enumerate(choices):
        letter = chr(ord('A') + i)
        options_formatted += f"\n({letter}) {choice}"
    try:
        correct_choice =  chr(ord('A') + choices.index(answer))
    except:
        print("Id", id)
        print("Choices", choices, type(choices[0]))
        print("Answer", answer, type(answer))
        raise ValueError
    if len(choices) not in STATS_BY_NUM_CHOICES:
        STATS_BY_NUM_CHOICES[len(choices)] = [0] * len(choices)
    STATS_BY_NUM_CHOICES[len(choices)][choices.index(answer)] += 1
    a = f'The answer is ({correct_choice}).'
    return (
        "\n".join([hint, question_formatted, options_formatted]),
        a,
    )

L = []
already_appeared_ids = set()
for (i, d) in enumerate(data):
    image, question, choices, answer = (
        d['image'],
        d['question'],
        d['choices'],
        d['answer'],
    )
    if 'id' in d:
        # print(d['id'])
        position_before_number = d['id'].rfind("-")
        id = int(d['id'][position_before_number + 1 : ])
        print(id)
    else:
        assert(False)
        id = i
    if id in BLOCKLIST:
        print("Id", id, "in blocklist")
        continue
    if id in already_appeared_ids:
        print("Id", id)
        assert(False)
    already_appeared_ids.add(id)
    # Format choices
    assert(choices is not None) # MCQ only for now
    if not isinstance(choices, list):
        assert(isinstance(choices, str))
        choices = ast.literal_eval(choices)
    choices = [str(c) for c in choices]
    if randomize_ordering:
        random.shuffle(choices)
    # Format answer
    if answer[0] == answer[-1] == "'":
        answer = answer[1:-1]
    elif answer[0] == answer[-1] =='"':
        answer = answer[1:-1]
    elif answer[0] == "'" and answer[-2:] == "'.":
        answer = answer[1:-2]
    elif answer[-1] == '.':
        answer = answer[:-1]
    elif answer.startswith("Answer: '"):
        answer = answer[len("Answer: '") : -1]
    elif answer.startswith('Answer: "'):
        answer = answer[len('Answer: "') : -1]
    elif answer.startswith('Answer: '):
        answer = answer[len('Answer: ') : ]
    elif answer.startswith("`Answer: "):
        answer = answer[len("`Answer: ") : -1]
    image = IMAGE_PATH + image
    q, a = generate_formatted_question_and_answer(image, question, choices, answer, id)
    # print(q)
    # print(a)
    text = '<ImageHere>' + q
    L += [
      {
        "id": id,
        "image": [image],
        "choices": choices,
        "answer": answer,
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

for k in sorted(STATS_BY_NUM_CHOICES.keys()):
    print(f"{sum(STATS_BY_NUM_CHOICES[k])} questions with {k} choices. Distribution is {STATS_BY_NUM_CHOICES[k]}.")

print(len(L), "samples generated.")

with open(LINE_CHARTS_PROCESSED_FILE_PATH, 'w') as file:
    json.dump(L, file, indent = 4)
