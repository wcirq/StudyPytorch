# -*- encoding: utf-8 -*-
"""
@File    : mian.py
@Time    : 2023/4/6 8:27
@Author  : wcirq
@Software: PyCharm
"""
import cv2
import numpy as np
import torch
from math import sqrt


def wrap(input, flow):
    """
    :param input:  B, C, H, W
    :param flow:  B, 2, H, W
    :return:
    """
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(input.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(input.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow
    # vgrid = grid

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output


if __name__ == '__main__':
    B, C, H, W = 1, 3, 128, 128



    interval = H / 8
    canvas = np.zeros((H, W, C), np.uint8)
    for i in range(1, int(H/interval)):
        cv2.line(canvas, (0, int(i*interval)), (W, int(i*interval)), np.random.randint(0, 255, (3,), dtype=int).tolist(), 1)
    for i in range(1, int(W/interval)):
        cv2.line(canvas, (int(i*interval), 0), (int(i*interval), W), np.random.randint(0, 255, (3,), dtype=int).tolist(), 1)

    canvas = cv2.imread(r"C:\Users\wcirq\Pictures\20220629111827.png")
    H, W, C = canvas.shape

    mean = canvas.mean()
    std = canvas.std()

    # inputs = torch.randn((B, C, H, W),)
    inputs = torch.from_numpy(canvas[None].astype(np.float32))
    inputs = inputs.permute(0, 3, 1, 2)
    inputs = (inputs-mean)/std

    flow = torch.randn((B, 2, H, W))

    diagonal_half = sqrt(W**2 + H**2) / 2
    center = (W/2, H/2)
    for r in range(H):
        for c in range(W):
            x_weight = c - center[0]
            y_weight = r - center[1]
            dist = (diagonal_half - sqrt((r-center[1])**2+(c-center[0])**2))/diagonal_half

            flow[0, 0, r, c] = x_weight * dist * 1.8
            flow[0, 1, r, c] = y_weight * dist * 0.8

    output = wrap(inputs, flow)

    output = output * std + mean
    output = output.permute(0, 2, 3, 1)

    dst = output.numpy().astype(np.uint8)[0]

    cv2.imshow("canvas", canvas)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)