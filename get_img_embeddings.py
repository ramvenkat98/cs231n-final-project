
# 16-bit inference for the 1.8B model

# Load Model
from my_utils import * 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import numpy as np
from tqdm import tqdm
import pickle

finetuned = True
if not finetuned:
    ckpt_path = "internlm/internlm-xcomposer2-vl-1_8b"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    print("About to load model")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    print("Loaded model")
    model.tokenizer = tokenizer
    filename = 'embeddings_for_unfinetuned_model.pkl'
else:
    path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32/finetune'
    filename = 'embeddings_for_pure_bar_chart_finetuned.pkl'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_400_sample_linechart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32/finetune'
    # filename = 'embeddings_for_pure_line_chart_finetuned.pkl'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_400_sample_barchart_and_linechart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32/finetune'
    # filename = 'embeddings_for_bar_chart_and_line_chart_finetuned.pkl'
    model = AutoPeftModelForCausalLM.from_pretrained(path_to_adapter, device_map="auto", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code = True)
    model.tokenizer = tokenizer

# Load MathVista Dataset
from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVista")

# Getting embeddings from testmini
classes_to_embeddings = {}
for d in tqdm(dataset["testmini"]):
    # if i % 10 == 0:
    #   print(i)
    if d['metadata']['context'] not in ('scatter plot', 'table', 'line plot', 'bar chart', 'geometry diagram'):
      continue
    context = d['metadata']['context']
    with torch.cuda.amp.autocast():
      with torch.no_grad():
        image = d['decoded_image'].convert('RGB')
        image = model.vis_processor(image).unsqueeze(0).cuda()
        img_embed = model.encode_img(image)
    if context not in classes_to_embeddings:
      classes_to_embeddings[context] = [img_embed]
    else:
      classes_to_embeddings[context].append(img_embed)


# Load our Line Chart and Bar Chart, Train and Val Datasets

classes_and_paths = {
    "Augmented Bar Charts (Val)": 'bar_charts_processed_visual_linguistic_val_randomized.json',
    "Augmented Line Charts (Val)": 'line_charts_processed_visual_linguistic_val_randomized.json',
    "Augmented Bar Charts (Train)": 'bar_charts_processed_visual_linguistic_train_randomized_llm_as_a_judge_460.json',
    "Augmented Line Charts (Train)": 'line_charts_processed_visual_linguistic_train_randomized_with_llm_as_a_judge_400.json',
}

for cls in classes_and_paths:
    path = classes_and_paths[cls]
    with open(path, 'r') as f:
        dataset = json.load(f)
    assert(cls not in classes_to_embeddings)
    classes_to_embeddings[cls] = []
    for d in dataset:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                image = d['image'][0]
                img_embed = model.encode_img(image)
                classes_to_embeddings[cls].append(img_embed)

# Print some info
for x in classes_to_embeddings.keys(): print(x, len(classes_to_embeddings[x]))

classes_to_embeddings_means = {
    x: [
        np.array(torch.mean(torch.squeeze(classes_to_embeddings[x][i]).cpu(), axis = 0)) for i in range(len(classes_to_embeddings[x]))
    ] for x in classes_to_embeddings
}

# Print more info
for x in classes_to_embeddings_means.keys(): print(x, len(classes_to_embeddings[x]))
print(classes_to_embeddings_means['bar chart'][0].shape)

with open(filename, 'wb') as f:
    pickle.dump(classes_to_embeddings_means, f)

print(f"Saved to {filename}\n")
