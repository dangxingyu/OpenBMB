# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import json

from collections import OrderedDict
from transformers import BartForConditionalGeneration, BartConfig
from model_center.model.config import BartConfig as myConfig
from tqdm import tqdm

base_path = '/home/zhaoweilin/dxy/ModelCenter'
# base_path = ''

def convert_model(version : str):
    config : BartConfig = BartConfig.from_pretrained(version)

    num_layers = config.num_hidden_layers
    bart = BartForConditionalGeneration.from_pretrained(version)
    # bart = torch.load(base_path)
    dict = bart.state_dict()
    keys = list(dict.keys())
    for key in keys:
        if key[:5] == 'model':
            new_key = key[6:]
            dict[new_key] = dict.pop(key)
    new_dict = OrderedDict()

    new_dict['input_embedding.weight'] = dict['shared.weight']
    new_dict['enc_position_embedding.weight'] = dict['encoder.embed_positions.weight']
    for i in tqdm(range(num_layers)):
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_q.weight'] = dict['encoder.layers.' + str(i) + '.self_attn.q_proj.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_q.bias'] = dict['encoder.layers.' + str(i) + '.self_attn.q_proj.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_k.weight'] = dict['encoder.layers.' + str(i) + '.self_attn.k_proj.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_k.bias'] = dict['encoder.layers.' + str(i) + '.self_attn.k_proj.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_v.weight'] = dict['encoder.layers.' + str(i) + '.self_attn.v_proj.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_v.bias'] = dict['encoder.layers.' + str(i) + '.self_attn.v_proj.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.weight'] = dict['encoder.layers.' + str(i) + '.self_attn.out_proj.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.bias'] = dict['encoder.layers.' + str(i) + '.self_attn.out_proj.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.weight'] = dict['encoder.layers.' + str(i) + '.self_attn_layer_norm.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.bias'] = dict['encoder.layers.' + str(i) + '.self_attn_layer_norm.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.weight'] = dict['encoder.layers.' + str(i) + '.fc1.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.bias'] = dict['encoder.layers.' + str(i) + '.fc1.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.weight'] = dict['encoder.layers.' + str(i) + '.fc2.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.bias'] = dict['encoder.layers.' + str(i) + '.fc2.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.weight'] = (dict['encoder.layernorm_embedding.weight'] if i == 0 
                                                                       else dict['encoder.layers.' + str(i - 1) + '.final_layer_norm.weight'])
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.bias'] = (dict['encoder.layernorm_embedding.bias'] if i == 0
                                                                       else dict['encoder.layers.' + str(i - 1) + '.final_layer_norm.bias'])
    new_dict['encoder.output_layernorm.weight'] = dict['encoder.layers.' + str(num_layers - 1) + '.final_layer_norm.weight']
    new_dict['encoder.output_layernorm.bias'] = dict['encoder.layers.' + str(num_layers - 1) + '.final_layer_norm.bias']

    new_dict['dec_position_embedding.weight'] = dict['decoder.embed_positions.weight']

    for i in tqdm(range(num_layers)):
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.project_q.weight'] = dict['decoder.layers.' + str(i) + '.self_attn.q_proj.weight']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.project_q.bias'] = dict['decoder.layers.' + str(i) + '.self_attn.q_proj.bias']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.project_k.weight'] = dict['decoder.layers.' + str(i) + '.self_attn.k_proj.weight']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.project_k.bias'] = dict['decoder.layers.' + str(i) + '.self_attn.k_proj.bias']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.project_v.weight'] = dict['decoder.layers.' + str(i) + '.self_attn.v_proj.weight']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.project_v.bias'] = dict['decoder.layers.' + str(i) + '.self_attn.v_proj.bias']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.attention_out.weight'] = dict['decoder.layers.' + str(i) + '.self_attn.out_proj.weight']
        new_dict['decoder.layers.' + str(i) + '.self_att.self_attention.attention_out.bias'] = dict['decoder.layers.' + str(i) + '.self_attn.out_proj.bias']
        new_dict['decoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.weight'] = dict['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.weight']
        new_dict['decoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.bias'] = dict['decoder.layers.' + str(i) + '.encoder_attn_layer_norm.bias']
        new_dict['decoder.layers.' + str(i) + '.ffn.ffn.w_in.w.weight'] = dict['decoder.layers.' + str(i) + '.fc1.weight']
        new_dict['decoder.layers.' + str(i) + '.ffn.ffn.w_in.w.bias'] = dict['decoder.layers.' + str(i) + '.fc1.bias']
        new_dict['decoder.layers.' + str(i) + '.ffn.ffn.w_out.weight'] = dict['decoder.layers.' + str(i) + '.fc2.weight']
        new_dict['decoder.layers.' + str(i) + '.ffn.ffn.w_out.bias'] = dict['decoder.layers.' + str(i) + '.fc2.bias']
        new_dict['decoder.layers.' + str(i) + '.self_att.layernorm_before_attention.weight'] = (dict['decoder.layernorm_embedding.weight'] if i == 0 
                                                                       else dict['decoder.layers.' + str(i - 1) + '.final_layer_norm.weight'])
        new_dict['decoder.layers.' + str(i) + '.self_att.layernorm_before_attention.bias'] = (dict['decoder.layernorm_embedding.bias']     if i == 0
                                                                       else dict['decoder.layers.' + str(i - 1) + '.final_layer_norm.bias'])
        
        new_dict['decoder.layers.' + str(i) + '.cross_att.layernorm_before_attention.weight'] = dict['decoder.layers.' + str(i) + '.self_attn_layer_norm.weight']
        new_dict['decoder.layers.' + str(i) + '.cross_att.layernorm_before_attention.bias'] = dict['decoder.layers.' + str(i) + '.self_attn_layer_norm.bias']
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.project_q.weight"] = dict['decoder.layers.' + str(i) + ".encoder_attn.q_proj.weight"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.project_k.weight"] = dict['decoder.layers.' + str(i) + ".encoder_attn.k_proj.weight"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.project_v.weight"] = dict['decoder.layers.' + str(i) + ".encoder_attn.v_proj.weight"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.attention_out.weight"] = dict['decoder.layers.' + str(i) + ".encoder_attn.out_proj.weight"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.project_q.bias"] = dict['decoder.layers.' + str(i) + ".encoder_attn.q_proj.bias"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.project_k.bias"] = dict['decoder.layers.' + str(i) + ".encoder_attn.k_proj.bias"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.project_v.bias"] = dict['decoder.layers.' + str(i) + ".encoder_attn.v_proj.bias"]
        new_dict['decoder.layers.' + str(i) + ".cross_att.self_attention.attention_out.bias"] = dict['decoder.layers.' + str(i) + ".encoder_attn.out_proj.bias"]

    new_dict['decoder.output_layernorm.weight'] = dict['decoder.layers.' + str(num_layers - 1) + '.final_layer_norm.weight']
    new_dict['decoder.output_layernorm.bias'] = dict['decoder.layers.' + str(num_layers - 1) + '.final_layer_norm.bias']
    # for k,v in new_dict.items():
    #     new_dict[k]=new_dict[k].half()

    new_dict['output_projection.weight'] = dict['lm_head.weight']
    new_dict['output_projection.bias'] = dict['final_logits_bias']
    torch.save(new_dict, os.path.join(base_path, 'configs', 'bart', 'bart-base', 'pytorch_model.pt'))
    
if __name__ == "__main__":
    version_list = ['facebook/bart-base']
    # for version in version_list:
        # convert_model(version)
    convert_model(version_list[0])
