#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: config.py
@time: 2023/3/30 下午3:24
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse
import yaml

__all__ = ['load_config']

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", type=str, default="", help="")

    args = parser.parse_args()
    return args

def load_config(file):
    assert (os.path.exists(file))

    config = None
    ##
    with open(file, 'r') as fin:
        config = yaml.safe_load(fin)

    return config

def main():
    args = GetArgs()


if __name__ == '__main__':
    main()
