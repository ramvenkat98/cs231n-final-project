import json
import re

with open('bar_charts_visual_linguistic_train.json', 'r') as f:
    L = json.load(f)

'''
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
'''

# Copied from the IPython notebook
def extract_code(response):
    # Look for code blocks delimited by ```python and ```
    code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = code_pattern.findall(response)
    # Join all matches into a single string of code
    return '\n'.join(matches).strip()

# Generate prompts for LLM as judge
prompts = []
for x in L:
    if 'code' not in x:
        assert('gpt_response' in x)
        x['code'] = extract_code(x['gpt_response'])
    prompts.append(f"Question Text:\n{x['question']}\nFigure Code:\n```\n{x['code']}\n```\nChoices:{x['choices']}\nStudent's Response:{x['answer']}\n")

with open('llm_as_judge_prompts.json', 'w') as f:
    json.dump(prompts, f, indent = 4)

from openai import OpenAI
client = OpenAI()

corrects, wrongs, unknowns = [], [], []

for i in range(len(prompts)):
    prompt_messages = [
        {
          "role": "system",
          "content": "You are grading a student's responses on a multiple choice test which requires the student to interpret and analyze visual figures. You are given the textual question and the code to generate the accompanying figure to the question. You are also given the student's response. Please explain your reasoning and grade the student's response as being either correct or wrong. End your answer with either \\boxed{correct} or \\boxed{wrong}.",
        },
        {
          "role": "user",
          "content": prompts[i]
        },
    ]
    # if L[i]['id'] == 'bar_charts-bar_charts_visual_linguistic-train-31':
    if i <= 1000:
        print(prompts[i])
        print(prompt_messages)
        response = client.chat.completions.create(
              model="gpt-4",
              messages=prompt_messages
        )
        text = response.choices[0].message.content
        pattern = r"\\boxed{([^}]*)}"
        print(text)
        matches = re.findall(pattern, text)
        if matches:
            last_match = matches[-1]  # Get the last item from the list of matches
            print(last_match)
        else:
            last_match = 'unknown'
        if last_match == 'correct':
            corrects.append(L[i]['id'])
        elif last_match == 'wrong':
            wrongs.append(L[i]['id'])
        else:
            unknowns.append(L[i]['id'])

print(corrects, wrongs, unknowns)

with open('llm_as_judge_results.json', 'w') as f:
    json.dump({"corrects": corrects, "wrongs": wrongs, "unknowns": unknowns}, f, indent = 4)
