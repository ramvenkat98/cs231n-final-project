import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import os
from PIL import Image

import torch
import transformers
from accelerate.utils import DistributedType
from data_mix import Mix_dataset
# from deepspeed import zero
# from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers import Trainer # , deepspeed
from transformers.trainer_pt_utils import LabelSmoother
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from transformers import Trainer

from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    LoreftIntervention,
    ReftDataCollator,
    ReftSupervisedDataset,
    ReftModel,
    dataset,
)
from tqdm import tqdm
import time
import pickle

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

model_name_or_path = 'internlm/internlm-xcomposer2-vl-1_8b'
max_length = 4096
device_map = None

intervenable_model_path = 'test_reft_vlm_debug_v6/checkpoint-18' # 'test_reft_vlm_v4/checkpoint-864'
rank = 64 # 32
layers_desc = 'all'
first_n = 9 # 5
last_n = 9 # 5

filename = 'model_eval_1_8b_reft_bar_chart_val_set_v4.pkl'
# line_chart = True

def get_dataset(line_chart):
    if line_chart:
        val_dataset_path = '../../../line_charts_processed_visual_linguistic_train_randomized_with_llm_as_a_judge_400_prefix_fixed.json'
        # '../../../line_charts_processed_visual_linguistic_val_randomized.json'
    else:
        val_dataset_path = '../../../bar_charts_processed_visual_linguistic_val_randomized.json'
    with open(val_dataset_path, 'r') as f:
        val_dataset = json.load(f)
    return val_dataset

def get_model():
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=None, # training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = max_length

    # Load model and tokenizer
    print(f'Load model from: {model_name_or_path}')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=None, #training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None, #training_args.cache_dir,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer

    layers = [i for i in range(model.config.num_hidden_layers)] if layers_desc == 'all' else -1
    representations = [{
        "layer": l, "component": f"model.layers[{l}].output",
        # this is needed for loading although dummy.
        "low_rank_dimension": rank,
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=rank,
        )
    } for l in layers]

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config)
    reft_model.load_intervention(f'{intervenable_model_path}/intervenable_model', include_model=True)
    reft_model.set_device("cuda")
    reft_model.print_trainable_parameters()
    for name, parameter in reft_model.named_parameters():
        parameter.data = parameter.data.to(dtype=torch.float32)
        parameter.requires_grad = False
    return model, reft_model, reft_config, tokenizer

def eval_encode(model, image, text):
    to_regress_embeds, attention_mask, targets, im_mask = model.interleav_wrap(
        image, text)
    inputs_embeds = to_regress_embeds[:, :model.max_length]
    attention_mask = attention_mask[:, :model.max_length]
    targets = targets[:, :model.max_length]
    im_mask = im_mask[:, :model.max_length].bool()
    labels = targets
    return (
        inputs_embeds.detach().cpu(),
        attention_mask.detach().cpu(),
        targets.detach().cpu(),
        im_mask.detach().cpu(),
        labels.detach().cpu(),
    )

# Get Model Outputs
def evaluate(reft_model, model, tokenizer, val_dataset, layers):
    final = {}
    i = 0
    start_time = time.time()
    for d in tqdm(val_dataset):
        # if i % 10 == 0:
        #   print(i)
        if d['id'] != 33:
            continue
        q = d['conversations'][0]['value']
        print("Query is", q)
        text = f"[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"
        image = [model.vis_processor(Image.open(d['image'][0]).convert('RGB')).unsqueeze(0).cuda()]
        # image = d['image'][0]
        # image = d['decoded_image'].convert('RGB')
        # image = model.vis_processor(image).unsqueeze(0).cuda()
        inputs_embeds, attention_mask, targets, im_mask, labels = eval_encode(model, image, text)
        last_position = inputs_embeds.shape[1] - 1
        intervention_locations = dataset.get_intervention_locations(
            last_position=last_position, first_n=first_n,
            last_n=last_n, pad_mode='first', num_interventions=len(layers),
        )
        intervention_locations = torch.IntTensor([intervention_locations])
        inputs = {
            'inputs_embeds': inputs_embeds.cuda(),
            'attention_mask': attention_mask.cuda(),
            # 'targets': targets.cuda(),
            'im_mask': im_mask.cuda(),
            'labels': labels.cuda(),
            # 'intervention_locations': torch.IntTensor([intervention_locations]),
            # 'id': torch.IntTensor([0]),
        }
        unit_locations = None
        assert(intervention_locations.dim() == 3)
        unit_locations={"sources->base": (
            None,
            intervention_locations.permute(1, 0, 2).tolist()
        )}
        base_outputs, counterfactual_outputs = reft_model.generate(
            inputs,
            sources = None,
            unit_locations = unit_locations,
            source_representations = None,
            intervene_on_prompt = True,
            subspaces = None,
            streamer = None,
            max_new_tokens = 1024,
            do_sample = False,
            temperature = 1.0,
            top_p = 0.8,
            repetition_penalty =1.005,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0],
            ],
            output_original_output = True,
        )
        responses = []
        for (i, outputs) in enumerate((base_outputs, counterfactual_outputs)):
            def get_response_from_outputs(query, outputs):
                outputs = outputs[0].cpu().tolist()
                response = tokenizer.decode(outputs, skip_special_tokens=True)
                response = response.split('[UNUSED_TOKEN_145]')[0]
                history = [] + [(query, response)]
                return response, history
            response, history = get_response_from_outputs(q, outputs)
            responses.append(response)
            if i == 0:
                print("Base response is", response)
            else:
                print("REFT response is", response)
        final[d['id']] = responses
    end_time = time.time()
    print(end_time - start_time)
    return final

def save(final, filename):
# Save Model Outputs
    with open(filename, 'wb') as file:
        pickle.dump(final, file)
