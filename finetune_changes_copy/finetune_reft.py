# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import os

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

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import pyvene as pv
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
import eval_finetuned_reft_on_val_set

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEBUG = False # True # False in v4
END_OF_PROMPT_STR = '[UNUSED_TOKEN_146]assistant\n'

rank = 32 # 32 in v4
batch_size = 1
img_size = 490
hd_num = -1
data_path = '/nlp/scr/ram1998/cs231n-final-project/data_bar_charts_and_line_charts_new.txt'
local_rank = 0
position = "f11+l11" # "f5+l5" in v4
seed = 42
given_num = False
share_weights = True
stop_at = None # None in v4
output_dir = 'test_reft_vlm_debug_v10'
save_directory = 'output_1_8b_lora_on_bar_and_line_chart_new_reft_try_10_debug_rank_32_f11l11_fixed_end_tokens_5_epochs/finetune'
model_max_length = 4096
model_name_or_path = 'internlm/internlm-xcomposer2-vl-1_8b'
max_length = 4096
device_map = None
num_train_epochs = 5 # 1

class ReftMixDataset:
    # Note that this is not entirely correct - we have not added a padding token at the start when we should.
    def __init__(self, model, mix_dataset, stop_at=None, **kwargs):
        self.result = []

        # get the data into the input format
        self.preprocess_and_tokenize(model, mix_dataset, stop_at)
        # compute intervention positions
        self.first_n, self.last_n = dataset.parse_positions(kwargs["position"])
        self.pad_mode = "first"
        for i, data_item in enumerate(self.processed_data_items):
            if stop_at is not None and i > stop_at:
                break
            intervention_locations = self.get_intervention_locations(last_position=self.last_positions[i], first_n=self.first_n, 
            last_n=self.last_n, pad_mode=self.pad_mode, **kwargs)
            data_item["intervention_locations"] = torch.IntTensor([intervention_locations])
            data_item["id"] = torch.IntTensor([i])
            self.result.append(data_item)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.result[i])
        
    def get_intervention_locations(self, **kwargs):
        return dataset.get_intervention_locations(**kwargs)
    
    def compute_intervention_and_subspaces(self, i: int, result: dict, last_position: int, **kwargs):
        raise NotImplementedYetError
        # compute intervention locs
        intervention_locations = self.get_intervention_locations(
            last_position=last_position,
            first_n=self.first_n, 
            last_n=self.last_n,
            pad_mode=self.pad_mode,
            **kwargs
        )
        result["intervention_locations"] = intervention_locations
        result["id"] = i
        return result
        
        # we should ideally add a padding token at the start so that things work out
        '''
        # add a single padding token BEFORE input_ids and fix everything
        if self.pad_mode == "first":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((torch.tensor([IGNORE_INDEX,]), result[field]))
                else:
                    result[field] = torch.cat((torch.tensor([self.tokenizer.pad_token_id,]), result[field]))
            result["intervention_locations"] = (torch.IntTensor(result["intervention_locations"]) + 1).tolist()
        elif self.pad_mode == "last":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((result[field], torch.tensor([IGNORE_INDEX,])))
                else:
                    result[field] = torch.cat((result[field], torch.tensor([self.tokenizer.pad_token_id,])))
        
        # attention masks
        if len(self.fields_to_mask) == 1:
            result["attention_mask"] = (result[self.fields_to_mask[0]] != self.tokenizer.pad_token_id).int()
        else:
            for field in self.fields_to_mask:
                result[f"{field}_mask"] = (result[field] != self.tokenizer.pad_token_id).int()
        '''

        # don't look at subspaces for now
        '''
        # subspaces
        if "subspaces" in data_item:
            num_interventions = kwargs["num_interventions"]
            share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
            if share_weights:
                num_interventions = num_interventions // 2
            # we now assume each task has a constant subspaces
            _subspaces = [data_item["subspaces"]] * num_interventions
            result["subspaces"] = _subspaces
        '''

    def preprocess_and_tokenize(self, model, mix_dataset, stop_at):
        self.processed_data_items = []
        self.last_positions = []
        print("Dataset length is", len(mix_dataset))
        if stop_at is None:
            stop_at = len(mix_dataset)
        for i, data_item in enumerate(mix_dataset):
            if DEBUG:
                print("Data item number", i)
            if i > stop_at:
                break
            # do the old collation logic here
            samples_before_collation = data_item['samples']
            assert('image' in samples_before_collation)
            assert(samples_before_collation['data_type'] == 'multi')
            samples = {}
            samples['text_input'] = [samples_before_collation['text_input']]
            samples['data_type'] = [samples_before_collation['data_type']]
            samples['image'] = [samples_before_collation['image']]

            text = samples['text_input']
            assert(samples['data_type'][0] == 'multi')
            image = samples['image']
            def encode(image, text):
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
            # encode text and image
            inputs_embeds, attention_mask, targets, im_mask, labels = encode(image, text)
            # find the last position to intervene on by encoding just the initial part
            assert(isinstance(text, list) and len(text) == 1 and isinstance(text[0], list) and len(text[0]) == 1)
            text_truncated = [[text[0][0][:text[0][0].find(END_OF_PROMPT_STR) + len(END_OF_PROMPT_STR)]]]
            inputs_embeds_truncated, _, _, _, _ = encode(image, text_truncated)
            last_position = inputs_embeds_truncated.shape[1] - 1
            self.last_positions.append(last_position)

            if DEBUG:
                print("Text is", text, "truncated is", text_truncated)
                print("Lengths are", inputs_embeds_truncated.shape, inputs_embeds.shape)
                print("Last position is", last_position)
            
            self.processed_data_items.append(
                {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': attention_mask,
                    'targets': targets,
                    'im_mask': im_mask,
                    'labels': labels,
                }
            )
        print("Done with preprocessing")

    def tokenize(self, data_item):
        # this function should not be needed
        return NotImplementedYetError

class ModifiedReftDataCollator:
    def __call__(self, instances):
        if DEBUG:
            print("Length of instances is", len(instances))
        assert(len(instances) == 1)
        batch_inputs = instances[0]
        max_seq_length = batch_inputs["inputs_embeds"].shape[-1]
        if DEBUG:
            print(max_seq_length, batch_inputs["intervention_locations"].shape)
        return batch_inputs

def make_supervised_data_module(model, layers, stop_at = None):
    print("Loading data...")
    if data_path.endswith('json'):
        train_json = json.load(open(data_path))
    elif data_path.endswith('txt'):
        train_json = {}
        with open(data_path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split(' ')
            with open(line[0]) as f:
                temp = json.load(f)
            if given_num:
                assert len(line) == 2
                num = int(float(line[1]) * 1000)
                if len(temp) > num:
                    temp = random.sample(temp, num)
                else:
                    ex_temp = []
                    for i in range(num - len(temp)):
                        ex_temp.append(random.choice(temp))
                    temp.extend(ex_temp)
            else:
                if len(line) == 2:
                    ratio = float(line[1])
                    new_len = int(len(temp) * ratio)
                    if ratio < 1:
                        temp = random.sample(temp, new_len)
                    elif ratio > 1:
                        ex_temp = []
                        for i in range(new_len - len(temp)):
                            ex_temp.append(random.choice(temp))
                        temp.extend(ex_temp)
            print(f'Load {len(temp)} samples from {line}')
            train_json[line[0]] = temp
    train_dataset = Mix_dataset(
        train_json,
        batch_size,
        img_size=img_size,
        hd_num=hd_num,
        local_rank=local_rank)
    print("Computed initial train dataset")
    train_dataset = ReftMixDataset(model, train_dataset, stop_at = stop_at,
        **{"num_interventions": len(layers), "position": position, 
           "share_weights": share_weights}
    )
    print("Computed REFT train dataset")
    print(str(len(train_dataset)) + 'samples is loaded')
    eval_dataset = None
    data_collator = ModifiedReftDataCollator()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

import pyvene as pv

class ReftTrainerForVLM(Trainer):
    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(
            save_directory=f"{output_dir}/intervenable_model", 
            include_model=True
        )

    def _load_best_model(self):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        unit_locations = None
        assert("intervention_locations" in inputs)
        if inputs["intervention_locations"].dim() == 3:
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
        if DEBUG:
            print(inputs)
            print(unit_locations)
        base_outputs, cf_outputs = intervenable(
            inputs,
            unit_locations=unit_locations,
            subspaces=None,
        )
        # return
        output = cf_outputs
        assert(cf_outputs is not None)
        if DEBUG:
            print("Output is", output)
            print("Loss is", output.loss)
            def get_response_from_output(output):
                # outputs = outputs[0].cpu().tolist()
                with torch.no_grad():
                    print("Labels is", inputs['labels'])
                    mask = inputs['labels'][0] != -100
                    correct_response = intervenable.model.tokenizer.decode(inputs['labels'][0][mask], skip_special_tokens=True)
                    print("Correct response is", correct_response)
                    shifted_mask = torch.cat([mask[1:], torch.tensor([True]).cuda()])
                    token_ids = torch.argmax(output.logits, dim = -1)[0]
                    response = intervenable.model.tokenizer.decode(token_ids[shifted_mask], skip_special_tokens=True)
                    response = response.split('[UNUSED_TOKEN_145]')[0]
                print("Response is", response)
                return response
            get_response_from_output(output)
        return (output, output) if return_outputs else output.loss

@dataclass
class ReftTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
    layers: str = field(
        default="all",
        metadata={"help": "Intervening layers."},
    )
    position: str = field(
        default="f1+l1",
        metadata={"help": "Intervening position string."},
    )
    share_weights: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    rank: int = field(default=1)
    max_n_train_example: int = field(default=None)

def train():
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

    if True: # training_args.fix_vit:
        model.vit.requires_grad_(False)
    else:
        model.vit.requires_grad_(True)
        model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity(
        )

    if True: # training_args.fix_sampler:
        model.vision_proj.requires_grad_(False)
    else:
        model.vision_proj.requires_grad_(True)

    if True:
        model.model.requires_grad_(False)

    if True:
        model.vit.resize_pos()

    # More Constants: TBD move to command line args
    layers = [i for i in range(model.config.num_hidden_layers)]

    # Initialize REFT model
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
    # TBD: Do we really need the lines below?
    for name, parameter in reft_model.named_parameters():
        parameter.data = parameter.data.to(dtype=torch.float32)
    reft_model.set_device("cuda")
    reft_model.print_trainable_parameters()

    data_module = make_supervised_data_module(model, layers, stop_at = stop_at)
    reft_training_arguments = ReftTrainingArguments(
        output_dir=output_dir,
        cache_dir=None,
        optim="adamw_torch",
        model_max_length=model_max_length,
        layers=layers,
        position=position,
        share_weights=share_weights,
        remove_unused_columns=False,
        rank=rank,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=num_train_epochs,
    )
    trainer = ReftTrainerForVLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=reft_training_arguments,
        **data_module,
    )
    trainer.train()
    trainer.save_state()
    reft_model.save(save_directory=save_directory)

    line_chart_dataset = eval_finetuned_reft_on_val_set.get_dataset(line_chart = True)
    final_val_results = eval_finetuned_reft_on_val_set.evaluate(reft_model, model, tokenizer, line_chart_dataset, layers)
    eval_finetuned_reft_on_val_set.save(final_val_results, 'debug_line_chart_validation.pkl')


    bar_chart_dataset = eval_finetuned_reft_on_val_set.get_dataset(line_chart = False)
    final_val_results = eval_finetuned_reft_on_val_set.evaluate(reft_model, model, tokenizer, bar_chart_dataset, layers)
    eval_finetuned_reft_on_val_set.save(final_val_results, 'debug_bar_chart_validation.pkl')
    # ReftModel.load(save_directory, model)
    '''
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)
    '''

if __name__ == '__main__':
    train()
