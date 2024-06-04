
# 16-bit inference for the 1.8B model

# Load Model
import json
from my_utils import * 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
finetuned = True
if not finetuned:
    ckpt_path = "internlm/internlm-xcomposer2-vl-1_8b"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True) # .cuda()
    print("About to load model")
    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    print("Loaded model")
    model.tokenizer = tokenizer
    filename = 'model_eval_on_visuolinguistic_val_set_1_8b.pkl'
else:
    from peft import AutoPeftModelForCausalLM
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_339_sample_barchart_fixed_vit/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_fixed_vit/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_530_sample_barchart_fixed_vit/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lr_times_2/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32/finetune'
    # path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_2/finetune'
    path_to_adapter = '/home/ramvenkat98/cs231n-final-project/InternLM-XComposer/finetune/output_1_8b_lora_on_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_2_learning_rate_divide_4/finetune'
    model = AutoPeftModelForCausalLM.from_pretrained(path_to_adapter, device_map="auto", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code = True)
    model.tokenizer = tokenizer
    # filename = 'model_eval_1_8b_finetuned.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_339_sample_barchart_fixed_vit.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judgefixed_vit.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judge_variable_vit.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lr_times_2.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_32.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_2.pkl'
    filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_460_sample_barchart_randomized_llm_as_judge_variable_vit_2_epochs_lora_2_learning_rate_divide_4.pkl'
    # filename = 'model_eval_1_8b_lora_on_visuolinguistic_val_set_530_sample_barchart_fixed_vit.pkl'
    print(f"Finetuned model at path {path_to_adapter}")

# Get Dataset
val_dataset_path = 'bar_charts_processed_visual_linguistic_val_randomized.json'
with open(val_dataset_path, 'r') as f:
    val_dataset = json.load(f)

# Get Model Outputs
from tqdm import tqdm
import time
final = {}
i = 0
start_time = time.time()
for d in tqdm(val_dataset):
    # if i % 10 == 0:
    #   print(i)
    '''
    if not finetuned:
        q = d['query']
        text = f"[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"
        image = d['decoded_image']
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                response = model_gen(model, text, image)
                final[d['pid']] = response
    else:
    '''
    text = d['conversations'][0]['value']
    image = d['image'][0] # Image.open(d['image'])
    # image = d['decoded_image'].convert('RGB')
    # image = model.vis_processor(image).unsqueeze(0).cuda()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(model.tokenizer, query=text, image=image, history=[], do_sample=False)
            final[d['id']] = response

end_time = time.time()
print(end_time - start_time)

# Save Model Outputs
import pickle
with open(filename, 'wb') as file:
    pickle.dump(final, file)
