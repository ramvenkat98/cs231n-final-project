import pickle
from my_utils import *

from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVista")

# filename = 'model_eval_1_8b.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_2_finetune.pkl'
# filename = 'model_eval_1_8b_finetuned.pkl'
# filename = 'output_1_8b_lora_on_530_sample_barchart_fixed_vit.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judgefixed_vit.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32_finetune.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_2_learning_rate_divide_4_finetune.pkl'
# filename = 'model_eval_1_8_b_lora_on_400_sample_line_chart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32.pkl'
filename = 'model_eval_1_8_b_lora_on_400_sample_bar_chart_and_line_chart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32.pkl'
# filename = 'model_eval_1_8b_lora_on_340_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32_finetune.pkl'
# filename = 'model_eval_1_8_b_lora_on_400_sample_bar_chart_and_line_chart_unrandomized_llm_as_judge_variable_vit_2_epochs_lora_32.pkl'
# filename = 'model_eval_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_8_finetune.pkl'
# filename = 'model_eval_1_8b_dora_on_460_sample_barchart_and_linechart_randomized_llm_as_judge_variable_vit_2_epochs_dora_32_finetune.pkl'
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

bar_correct, bar_unextractable, bar_total = stats_by_question_type_and_context_type[('bar chart', 'multi_choice')]
print("Bar Chart:", bar_correct, bar_unextractable, bar_total, 100.0 * bar_correct / bar_total)
line_correct, line_unextractable, line_total = stats_by_question_type_and_context_type[('line plot', 'multi_choice')]
print("Line Plot:", line_correct, line_unextractable, line_total, 100.0 * line_correct / line_total)

total_correct, total_unextractable, total_total = 0, 0, 0

for x in stats_by_question_type_and_context_type:
    diagram, qn_type = x
    if qn_type == 'multi_choice' and (diagram != 'bar chart' and diagram != 'line plot'):
        correct, unextractable, total = stats_by_question_type_and_context_type[x]
        total_total += total
        total_unextractable += unextractable
        total_correct += correct

print("Other Charts:", total_correct, total_unextractable, total_total, 100.0 * total_correct / total_total)
