# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
export network to infer `AIR` backend.
"""

import argparse
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.config import mnist_cfg as cfg
from src.lenet import LeNet5


parser = argparse.ArgumentParser(description='MindSpore MNIST Example')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default="",
                    help='if mode is test, must provide path where the trained ckpt file')
args = parser.parse_args()


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    # define fusion network
    network = LeNet5(cfg.num_classes)
    # load network checkpoint
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.ones([1, 1, cfg.image_height, cfg.image_width]), mindspore.float32)
    export(network, inputs, file_name=cfg.air_name, file_format='AIR')
