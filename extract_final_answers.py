import pickle
from my_utils import *

from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVista")

# filename = 'model_eval_1_8b.pkl'
# filename = 'model_eval_1_8b_finetuned.pkl'
# filename = 'output_1_8b_lora_on_530_sample_barchart_fixed_vit.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judgefixed_vit.pkl'
filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit.pkl'
with open(filename, 'rb') as f:
  eval_1_8 = pickle.load(f)

# Just get question and answer type counts, for reference
from collections import Counter
question_and_answer_type_counts = {}
for x in dataset['testmini']:
  tup = x['question_type'], x['answer_type'], x['metadata']['context'], x['metadata']['category']
  if tup not in question_and_answer_type_counts:
    question_and_answer_type_counts[tup] = 1
  else:
    question_and_answer_type_counts[tup] += 1

# Extract answers
full_out_1_8b = {}
i = 0
for d in tqdm(dataset["testmini"]):
    res = extract_answer(eval_1_8[d['pid']], d, quick_extract = True)
    # print("Response is", res)
    d['extraction'] = res
    full_out_1_8b[d['pid']] = (
        d['question_type'],
        d['answer_type'],
        d['metadata']['context'],
        d['metadata']['grade'],
        # d['metadata']['category'],
        d['metadata']['task'],
        res,
        d['choices'],
        d['answer'],
        d['answer'] if d['question_type'] != 'multi_choice' else chr(ord('A') + d['choices'].index(d['answer']))
    )

stats_by_question_type_and_context_type = {}
i = 0
for x in full_out_1_8b:
  # if i >= 50:
  #   break
  # i += 1
  question_type, _, context, grade, task, chosen_option, _, _, correct_option = full_out_1_8b[x]
  # print(question_type, chosen_option, correct_option)
  key = (context, question_type)
  if key not in stats_by_question_type_and_context_type:
    stats_by_question_type_and_context_type[key] = (int(chosen_option == correct_option), int(chosen_option == ''), 1)
  else:
    stats_by_question_type_and_context_type[key] = (
        stats_by_question_type_and_context_type[key][0] + int(chosen_option == correct_option),
        stats_by_question_type_and_context_type[key][1] + int(chosen_option == ''),
        stats_by_question_type_and_context_type[key][2] + 1
    )

print(stats_by_question_type_and_context_type)
