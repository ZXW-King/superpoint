#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2023/3/9 下午6:32
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse
from model.export.superpoint_bn import SuperPointBNNet
import torch
from utils.config import load_config

W, H = 640, 400

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--output", type=str, help="output model path")
    parser.add_argument("--config", type=str, help="config yaml file")

    args = parser.parse_args()
    return args

def main():
    args = GetArgs()

    config = load_config(args.config)
    model = SuperPointBNNet(config['model'], using_bn=config['model']['using_bn'])

    # load ckpts
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # device = torch.device("cuda")
    # model.to(device)
    model.eval()

    onnx_input = torch.rand(1, 1, H, W)
    onnx_input = onnx_input.to("cpu")
    torch.onnx.export(model,
                      onnx_input,
                      args.output,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output1', 'output2'])


if __name__ == '__main__':
    main()
