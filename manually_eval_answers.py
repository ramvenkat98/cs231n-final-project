import pickle

from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVista")

filename = 'model_eval_1_8b.pkl'
# filename = 'model_eval_1_8b_finetuned.pkl'
# filename = 'model_eval_1_8b_lora_on_339_sample_barchart_fixed_vit.pkl'
with open(filename, 'rb') as f:
  eval_1_8 = pickle.load(f)

full_out_1_8b = {}
i = 0
for d in dataset["testmini"]:
    res = eval_1_8[d['pid']]
    if d['question_type'] == 'multi_choice' and d['metadata']['context'] == 'bar chart':
        print(d['query'])
        print("Response:", res)
        print("Answer:", d['answer'])
