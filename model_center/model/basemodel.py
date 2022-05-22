# coding=utf-8
# Copyright 2022 The OpenBMB team.
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
from typing import Union
import torch
import bmtrain as bmt
from .config.config import Config
from ..utils import check_web_and_convert_path

class BaseModel(torch.nn.Module):

    _CONFIG_TYPE = Config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config=None):
        if config is None:
            config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_name_or_path)
        path = check_web_and_convert_path(pretrained_model_name_or_path, 'model')
        model = cls(config)
        bmt.load(model, os.path.join(path, 'pytorch_model.pt'), strict=True)
        return model

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config = cls._CONFIG_TYPE.from_json_file(json_file)
        model = cls(config)
        bmt.init_parameters(model)
        return model
