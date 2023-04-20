# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2023/4/20 11:05
@Author  : wcirq
@Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_signal(fres=[5], sr=44100, total_time=1):
    total_time = total_time  # 秒
    amp = 1000  # 振幅
    phase = 0  # 相位
    stop = total_time * 2 * np.pi
    x = np.linspace(0, stop, sr*total_time+1)[:-1]
    y = amp * np.sin(fres[0]*x+phase)
    for fre in fres[1:]:
        y += amp * np.sin(fre*x+phase)
    return y/len(fres), sr


def main():
    sr = 30  # 采样率
    signal, _ = generate_signal([2, 4], sr=sr)

    # max_freq = int(sr / 2)  # 一般信号的采样率要大于最大频率的2倍
    max_freq = sr
    ratio = 2
    for freq in range(max_freq):
        temp, _ = generate_signal([freq], sr=sr)
        temp2, _ = generate_signal([freq], sr=ratio*sr)

        correlation = (signal * temp).sum()
        # if abs(correlation)<0.001:
        #     continue

        x = np.arange(0, len(temp2))

        plt.text(0, 1100, f"频率{freq} 相关性:{correlation:.2f}")
        plt.ylim(-1000, 1000)
        l1 = plt.scatter(x[::ratio], signal)
        l2 = plt.scatter(x[::ratio], temp)
        l3, = plt.plot(temp2, linestyle='--')
        plt.legend(handles=[l1, l2, l3], labels=['signal', 'temp', 'temp2'], loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    main()