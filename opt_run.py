from src.pipeedge.models.transformers.opt import OPTModelShard
import torch
from transformers import AutoTokenizer, OPTModel
from transformers import AutoConfig
import os, sys
from src.pipeedge.models import ModuleShard, ModuleShardConfig
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = OPTModel.from_pretrained("facebook/opt-350m")


model_file = 'OPT_350M'

if(not os.path.isfile('./' + model_file + '.npz')): 
    state_dict = model.state_dict()
    weights = {}
    for key, val in state_dict.items():
        weights[key] = val
    np.savez(model_file, **weights)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

config = AutoConfig.from_pretrained("facebook/opt-350m")
layer_start = 1
layer_end = 96
is_first = layer_start == 1
is_last = layer_end == 96
shard_config = ModuleShardConfig(layer_start=layer_start, layer_end=layer_end, is_first=is_first, is_last=is_last)
opt_model_shard = OPTModelShard(config, shard_config, "OPT_350M.npz")
outputs1 = opt_model_shard.forward(inputs.input_ids, inputs.attention_mask)

print(outputs[0] == outputs1[0])
