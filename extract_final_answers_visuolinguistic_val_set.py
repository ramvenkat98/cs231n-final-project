import pickle
from my_utils import *

from tqdm import tqdm

# filename = 'model_eval_on_visuolinguistic_val_set_1_8b.pkl'
# filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_530_sample_barchart_fixed_vit.pkl'
# filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_339_sample_barchart_fixed_vit.pkl'
# filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judgefixed_vit.pkl'
filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judge_variable_vit.pkl'

with open(filename, 'rb') as f:
  eval_1_8 = pickle.load(f)

'''
val_data_filename = 'bar_charts/bar_charts_visual_linguistic_test.json'
with open(val_data_filename, 'r') as f:
    val_data = json.load(f)
'''

val_data_processed_filename = 'bar_charts_processed_visual_linguistic_val_randomized.json'
with open(val_data_processed_filename, 'r') as f:
    val_data_processed = json.load(f)

'''
val_data_processed_id_to_contents = {}
for x in val_data_processed:
    val_data_processed_id_to_contents[x['id']] = x
print(val_data_processed_id_to_contents.keys())
'''

# Extract answers
full_out_1_8b = {}
i = 0
print(eval_1_8)
for d in tqdm(val_data_processed):
    id = d['id'] # int(d['id'][d['id'].rfind('-') + 1 : ])
    if id not in eval_1_8:
        print(f"Id {d['id']} not present, must be in blocklist")
        continue
    d['question_type'] = 'multi_choice'
    d['answer_type'] = '-'
    d['query'] = d['conversations'][0]['value']
    res = extract_answer(eval_1_8[id], d, quick_extract = True)
    # print("Response is", res)
    d['extraction'] = res
    print(id, d['answer'], d['extraction'], d['choices'])
    full_out_1_8b[id] = (
        d['question_type'],
        d['answer_type'],
        None, # d['metadata']['context'],
        None, #d['metadata']['grade'],
        # d['metadata']['category'],
        None, # d['metadata']['task'],
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
