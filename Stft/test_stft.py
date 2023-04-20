# -*- encoding: utf-8 -*-
"""
@File    : test_sftp.py
@Time    : 2023/4/3 15:53
@Author  : wcirq
@Software: PyCharm
"""
import wave

import numpy as np
import pyaudio
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
    x = np.linspace(0, stop, sr*total_time+1)[:-1]
    y = amp * np.sin(fres[0]*x+phase)
    for fre in fres[1:]:
        y += amp * np.sin(fre*x+phase)
    return y/len(fres), sr


def play_autio(wave_datas=None, sampling_rate=48000, sampwidth=2, nchannels=1):
    # wav_path = "dataset_raw/4aa104ef/common_voice_zh-CN_18646940.wav"
    # with wave.open(wav_path, "rb") as read_file:
    #     params = read_file.getparams()
    #     nchannels, sampwidth, sampling_rate, nframes = params[:4]
    #     data = read_file.readframes(nframes)
    #     wave_datas = np.fromstring(data, dtype=np.short)

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sampwidth),
                    channels=nchannels,
                    rate=sampling_rate,
                    output=True)

    # 还原音频并播放
    fream_len = sampling_rate
    side = int(np.ceil(wave_datas.size/fream_len))
    pad = fream_len*side - wave_datas.size
    wave_datas_pad = np.pad(wave_datas, (0, pad))
    wave_datas_pad = np.reshape(wave_datas_pad, (side, fream_len))
    for wave_data in wave_datas_pad:
        wave_data_bytes = wave_data.tobytes()
        stream.write(wave_data_bytes)  # 播放原始语音


def test_sftp(fre=5, pw=None):
    total_time = 5

    fres = []
    base_fre = 10000
    fres += [base_fre, base_fre*2, base_fre*3]
    base_fre = 10000
    fres += [base_fre, base_fre*2, base_fre*3]
    base_fre = 10000
    fres += [base_fre, base_fre*2, base_fre*3]

    y, sr = generate_autio(fres, total_time)
    y_norm = ((y-y.min())/(y.max()-y.min())-0.5)*2

    y_short = y_norm * int('1'*15, 2)
    y_short = np.asarray(y_short, np.short)
    play_autio(y_short, sampling_rate=sr)

    plt.figure("Image")  # 图像窗口名称
    plt.plot(y)
    plt.show()

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

    image = np.copy(spec.numpy()[:])
    image = cv2.resize(image, (len(_f0), image.shape[0]))
    image = (image-image.min())/(image.max()-image.min()) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, _f in enumerate(_f0):
        h = int(_f/ratio)
        w = i
        image[h, w, :] = [0, 0, 255]

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
