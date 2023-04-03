# -*- encoding: utf-8 -*-
"""
@File    : test_sftp.py
@Time    : 2023/4/3 15:53
@Author  : wcirq
@Software: PyCharm
"""
import numpy as np
import torch
import pyworld
import cv2
from matplotlib import pyplot as plt


def generate_autio(fres=[5], total_time=1):
    total_time = total_time  # 秒
    sr = 44100  # 采样率
    amp = 1000  # 振幅
    phase = 0  # 相位
    stop = total_time * 2 * np.pi
    x = np.linspace(0, stop, sr*total_time)
    y = amp * np.sin(fres[0]*x+phase)
    for fre in fres[1:]:
        y += amp * np.sin(fre*x+phase)
    return y/len(fres), sr

def test_sftp(fre=5, pw=None):
    total_time = 1
    base_fre = 600
    y, sr = generate_autio([base_fre, base_fre*2, base_fre*3], total_time)
    y_norm = (y-y.min()/(y.max()-y.min()))
    # plt.figure("Image")  # 图像窗口名称
    # plt.plot(x, y)
    # plt.show()
    # print()

    n_fft = 2048
    hop_size = 512
    win_size = 2048
    center = False
    y = torch.from_numpy(y)
    window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    # n_fft 默認是 2048，对应的 sample rate 是 22.05 KHz，即语谱图能表示的频率最高为22050
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)  # 复数的模
    ratio = 22050 / (n_fft/2)

    _f0, _time = pyworld.dio(y_norm, sr)  # 基本周波数の抽出
    f0 = pyworld.stonemask(y_norm, _f0, _time, sr)  # 基本周波数の修正

    image = np.copy(spec.numpy()[:100])
    image = cv2.resize(image, (len(_f0), image.shape[0]))
    image = (image-image.min())/(image.max()-image.min()) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, _f in enumerate(_f0):
        h = int(_f/ratio)
        w = i
        image[h, w, :]= [0, 0, 255]

    for i, f in enumerate(f0):
        h = int(f/ratio)
        w = i
        image[h, w, :]= [0, 255, 0]

    cv2.imshow("image", image)
    cv2.waitKey(0)
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(image)
    # plt.axis('on')  # 关掉坐标轴为 off
    # plt.title('image')  # 图像题目
    # plt.show()

    print()


if __name__ == '__main__':
    fre = 500
    test_sftp(fre)
