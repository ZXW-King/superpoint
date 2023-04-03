#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: test_onnx.py
@time: 2023/3/16 下午12:30
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from onnx.onnxmodel import ONNXModel
import cv2
import numpy as np
from demo_superpoint_orb import nms_fast
from file import Walk, MkdirSimple
from tqdm import tqdm

W, H = 640, 400
cell = 8
conf_thresh = 0.015
nms_dist = 4


__all__ = ['W', 'H', 'test_onnx']

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, help="")
    parser.add_argument("--model", type=str, help="")
    parser.add_argument("--output", type=str, help="")

    args = parser.parse_args()
    return args


def GetImages(path):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(path).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        paths = Walk(path, ['jpg', 'png', 'jpeg'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    return paths, root_len


def ProcessPoints(semi):
    # --- Process points.
    dense = np.exp(semi)  # Softmax.
    npsum = np.sum(dense, axis=0) + .00001
    dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / cell)
    Wc = int(W / cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.

    return pts


def ProcessDescribe(pts, coarse_desc):
    samp_pts = pts[:2, :].copy()
    samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
    samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
    samp_pts = samp_pts.transpose(0, 1).contiguous()
    samp_pts = samp_pts.view(1, 1, -1, 2)
    samp_pts = samp_pts.float()

    # desc = grid_sample(coarse_desc, samp_pts)
    desc = coarse_desc
    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

    return desc

def ShowPoint(image, pts):
    for y, x, confidence in pts.T:
        cv2.circle(image, (int(y), int(x)), 1, (0, 255, 0), -1, lineType=16)

    return image

def test_onnx(img_path, output_path, output_name, model_file):
    model = ONNXModel(model_file)
    img_org = cv2.imread(img_path)
    img_org = cv2.resize(img_org, (W, H), cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = np.expand_dims(img, axis=0).astype("float32")
    img = np.expand_dims(img, axis=0).astype("float32")

    output = model.forward(img)
    semi = output[0][0]
    pts = ProcessPoints(semi)
    img_show = ShowPoint(img_org, pts)

    name = os.path.splitext(output_name)[0] + ".png"
    output_file = os.path.join(output_path, name) if not os.path.isfile(output_path) else output_path
    MkdirSimple(output_file)

    cv2.imwrite(output_file, img_show)
    coarse_desc = output[1][0]
    # ProcessDescribe(pts, coarse_desc)

def main():
    args = GetArgs()
    files, root_len = GetImages(args.image)
    for f in tqdm(files):
        output_name = f[root_len + 1:]
        test_onnx(f, args.output, output_name, args.model)


if __name__ == '__main__':
    main()
