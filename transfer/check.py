import os
import torch
import json

from collections import OrderedDict
from transformers import BartForConditionalGeneration, BartConfig
from model_center.model.config import BartConfig as myConfig
from tqdm import tqdm

bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

dict = bart.state_dict()
keys = list(dict.keys())
keys.sort()
for key in keys:
    print(key)