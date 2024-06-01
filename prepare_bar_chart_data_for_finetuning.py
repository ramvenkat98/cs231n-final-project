import json
import ast
import random

randomize_ordering = True

BAR_CHARTS_ORIGINAL_JSON_FILE_PATH = '/home/ramvenkat98/cs231n-final-project/bar_charts/bar_charts_visual_linguistic_train.json'
# BAR_CHARTS_ORIGINAL_JSON_FILE_PATH = '/Users/ramvenkat98/Desktop/stanford_courses/cs231n/cs231n_final_project/bar_charts/bar_charts_visual_linguistic_train.json'
BAR_CHARTS_PROCESSED_FILE_PATH = 'bar_charts_processed_visual_linguistic_train_randomized.json'
IMAGE_PATH = '/home/ramvenkat98/cs231n-final-project/'
BLOCKLIST = [i for i in range(1, 26)] + [
    2, 5, 7, 8, 9, 15, 41, 66, 154, 164, 200, 202, 219, 237, 260, 278, 282, 289,
    301, 331, 0, 421, 434, 463, 494
]

with open(BAR_CHARTS_ORIGINAL_JSON_FILE_PATH, 'r') as file:
    data = json.load(file)

STATS_BY_NUM_CHOICES = {}

def generate_formatted_question_and_answer(image, question, choices, answer, id):
    hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    question_formatted = f"Question: {question}"
    options_formatted = "Choices:"
    if not isinstance(choices, list):
        assert(isinstance(choices, str))
        choices = ast.literal_eval(choices)
    choices = [str(c) for c in choices]
    if randomize_ordering:
        random.shuffle(choices)
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
        # print(id)
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
    assert(choices is not None) # MCQ only for now
    image = IMAGE_PATH + image
    q, a = generate_formatted_question_and_answer(image, question, choices, answer, id)
    # print(q)
    # print(a)
    text = '<ImageHere>' + q
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

for k in sorted(STATS_BY_NUM_CHOICES.keys()):
    print(f"{sum(STATS_BY_NUM_CHOICES[k])} questions with {k} choices. Distribution is {STATS_BY_NUM_CHOICES[k]}.")

print(len(L), "samples generated.")

with open(BAR_CHARTS_PROCESSED_FILE_PATH, 'w') as file:
    json.dump(L, file, indent = 4)
