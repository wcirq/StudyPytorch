# -*- encoding: utf-8 -*-
"""
@File    : 卡尔曼滤波.py
@Time    : 2023/7/28 16:39
@Author  : wcirq
@Software: PyCharm
@link    : https://www.kalmanfilter.net/CN/alphabeta_cn.html
"""
import numpy as np
from matplotlib import pyplot as plt

def main():
    """估计黄金重量"""
    x00 = 1000
    measurements = []
    estimate = []
    for i in range(99):
        z1 = np.random.randint(900, 1200)
        a1 = 1/(i+1)
        x10 = x00
        x11 = x10 + a1*(z1-x10)
        print(f"x{i+1},{i+1}: {z1} {x11}")
        measurements.append(z1)
        estimate.append(x11)
        x21 = x11

        x00 = x21

    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    plt.legend()
    plt.show()


def main2():
    """跟踪直线匀速运动的飞行器"""
    t = 5
    v = 40
    x0 = 30000

    x00 = x0 + t * v  # x^
    x_00 = v    # x˙^

    for i in range(99):
        z1 = x0 + v * (i+1) * t + np.random.randint(-5, 5)
        a1 = 0.2
        b1 = 0.1
        x10 = x00
        x_10 = x_00
        x11 = x10 + a1*(z1-x10)
        x_11 = x_10 + b1*((z1-x10)/t)

        x21 = x11 + t * x_11
        x_21 = x_11

        x00 = x21
        x_00 = x_21

        print(f"{z1:0.1f} {x11:0.1f} {x_11:0.1f}")




if __name__ == '__main__':
    # main()  # 估计黄金重量
    main2()  # 跟踪直线匀速运动的飞行器