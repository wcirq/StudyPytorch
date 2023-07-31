# -*- encoding: utf-8 -*-
"""
@File    : 卡尔曼滤波.py
@Time    : 2023/7/28 16:39
@Author  : wcirq
@Software: PyCharm
@link    : https://www.kalmanfilter.net/CN/alphabeta_cn.html
"""
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt


def main():
    """估计黄金重量"""
    x = 1000
    x00 = x
    actual = []
    measurements = []
    estimate = []
    for i in range(100):
        z1_true = x
        z1 = z1_true + np.random.randint(-150, 150)
        a1 = 1 / (i + 1)

        x10 = x00  # 上一刻对当前重量的预测 = 上一刻对上一刻的估计

        x11 = x10 + a1 * (z1 - x10)  # 当前对当前的估计

        print(f"x{i + 1},{i + 1}: {z1} {x11}")

        actual.append(z1_true)
        measurements.append(z1)
        estimate.append(x11)

        x21 = x11  # 当前对下一刻的预测 =

        x00 = x21  # 下一刻对下一刻的估计 =

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    plt.legend()
    plt.show()


def main2():
    """跟踪直线匀速运动的飞行器"""
    t = 5  # 间隔时间
    v = 40  # 速度
    x0 = 30000  # 初始距离

    x00 = x0 + t * v  # x^
    x_00 = v  # x˙^

    actual = []
    measurements = []
    estimate = []
    prediction = []
    for i in range(10):
        time = (i + 1) * t
        z1_true = x0 + v * time
        z1 = z1_true + np.random.randint(-100, 100)  # 当前的距离测量值
        a1 = 0.8
        b1 = 0.5

        x10 = x00  # 上一刻对当前的距离预测 = 上一刻对上一刻的距离估计
        x_10 = x_00  # 上一刻对当前的速度预测 = 上一刻对上一刻的速度估计

        x11 = x10 + a1 * (z1 - x10)  # 当前对当前距离的估计 =
        x_11 = x_10 + b1 * ((z1 - x10) / t)  # 当前对当前速度的估计 =

        x21 = x11 + t * x_11  # 当前对下一刻的距离预测 =
        x_21 = x_11  # 当前对下一刻的速度预测 =

        actual.append(z1_true)
        measurements.append(z1)
        estimate.append(x11)
        prediction.append(x10)

        x00 = x21  # 下一刻对下一刻的距离估计 = 当前对下一刻的距离预测
        x_00 = x_21  # 下一刻对下一刻的距离估计 = 当前对下一刻的速度预测

        print(f"{z1:0.1f} {x11:0.1f} {x_11:0.1f}")

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    plt.plot(prediction, label="prediction")
    plt.legend()
    plt.show()


def main3():
    """跟踪直线加速运动的飞行器"""
    t = 5  # 间隔时间
    v = 50  # 速度
    x0 = 30000  # 初始距离
    a = 8  # 加速度
    start_accelerate_time = 20  # 开始加速时间

    x00 = x0 + t * v  # x^
    x_00 = v  # x˙^
    x__00 = a  # x˙^

    actual = []
    measurements = []
    estimate = []
    prediction = []
    for i in range(10):
        time = (i + 1) * t
        if time > start_accelerate_time:
            # 开始加速
            z1_true = x0 + v * time + \
                      v * (time - start_accelerate_time) + (a * (time - start_accelerate_time) ** 2) / 2
            z1 = z1_true + np.random.randint(-300, 300)  # 当前的距离测量值
        else:
            # 匀速运动
            z1_true = x0 + v * time
            z1 = z1_true + np.random.randint(-300, 300)  # 当前的距离测量值
        a1 = 0.2
        b1 = 0.1
        x10 = x00  # 上一刻对当前的距离预测 = 上一刻对上一刻的距离估计
        x_10 = x_00  # 上一刻对当前的速度预测 = 上一刻对上一刻的速度估计

        x11 = x10 + a1 * (z1 - x10)  # 当前对当前距离的估计 =
        x_11 = x_10 + b1 * ((z1 - x10) / t)  # 当前对当前速度的估计 =

        x21 = x11 + t * x_11  # 当前对下一刻的距离预测 =
        x_21 = x_11  # 当前对下一刻的速度预测 =

        actual.append(z1_true)
        measurements.append(z1)
        estimate.append(x11)
        prediction.append(x10)

        x00 = x21  # 下一刻对下一刻的距离估计 = 当前对下一刻的距离预测
        x_00 = x_21  # 下一刻对下一刻的距离估计 = 当前对下一刻的速度预测

        print(f"{z1:0.1f} {x11:0.1f} {x_11:0.1f}")

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    plt.plot(prediction, label="prediction")
    plt.legend()
    plt.show()


def main4():
    """用 α−β−γ滤波器跟踪直线加速运动的飞行器"""
    t = 5  # 间隔时间
    v = 50  # 速度
    x0 = 30000  # 初始距离
    a = 8  # 加速度
    start_accelerate_time = 20  # 开始加速时间

    x00 = x0 + t * v  # x^
    x_00 = v  # x˙^
    x__00 = a  # x˙^

    actual = []
    measurements = []
    estimate = []
    prediction = []
    for i in range(10):
        time = (i + 1) * t
        if time > start_accelerate_time:
            # 开始加速
            z1_true = x0 + v * time + \
                      v * (time - start_accelerate_time) + (a * (time - start_accelerate_time) ** 2) / 2
            z1 = z1_true + np.random.randint(-300, 300)  # 当前的距离测量值
        else:
            # 匀速运动
            z1_true = x0 + v * time
            z1 = z1_true + np.random.randint(-300, 300)  # 当前的距离测量值
        a1 = 0.5
        b1 = 0.4
        r1 = 0.1
        x10 = x00  # 上一刻对当前的距离预测 = 上一刻对上一刻的距离估计
        x_10 = x_00  # 上一刻对当前的速度预测 = 上一刻对上一刻的速度估计
        x__10 = x__00  # 上一刻对当前的加速度预测 = 上一刻对上一刻的加速度估计

        x11 = x10 + a1 * (z1 - x10)  # 当前对当前距离的估计 =
        x_11 = x_10 + b1 * ((z1 - x10) / t)  # 当前对当前速度的估计 =
        x__11 = x__10 + r1 * ((z1 - x10) / (0.5 * t ** 2))  # 当前对当前加速度的估计 =

        x21 = x11 + t * x_11 + (x__11 * t ** 2) / 2  # 当前对下一刻的距离预测 =
        x_21 = x_11 + x__11 * t  # 当前对下一刻的速度预测 =
        x__21 = x__11  # 当前对下一刻的加速度预测 =

        actual.append(z1_true)
        measurements.append(z1)
        estimate.append(x11)
        prediction.append(x10)

        x00 = x21  # 下一刻对下一刻的距离估计 = 当前对下一刻的距离预测
        x_00 = x_21  # 下一刻对下一刻的距离估计 = 当前对下一刻的速度预测
        x__00 = x__21  # 下一刻对下一刻的距离估计 = 当前对下一刻的速度预测

        print(f"{z1:0.1f} {x11:0.1f} {x_11:0.1f}")

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    plt.plot(prediction, label="prediction")
    plt.legend()
    plt.show()


def main5():
    """估计大楼高度"""
    height_true = 50      # 大楼高度的真值是50米
    tool_error = 5        # 高度计误差（标准差）是5米
    height_eye = 60              # 肉眼估计高度
    std = 15              # 肉眼估计标准差
    var = std ** 2        # 肉眼估计方差

    x00 = height_eye
    p00 = var

    confidence = 0.95

    kalman_gain = []
    actual = []
    measurements = []
    estimate = []
    prediction = []
    low_bound = []
    high_bound = []

    z1s = [49.03, 48.44, 55.21, 49.98, 50.6, 52.61, 45.87, 42.64, 48.26, 55.84]

    for i in range(len(z1s)):
    # for i in range(100):
        x10 = x00
        p10 = p00

        z1 = z1s[i]
        # z1 = height_true + np.random.randint(-tool_error, tool_error)
        r1 = tool_error ** 2

        K1 = p10 / (p10+r1)             # 卡尔曼增益
        x11 = x10 + K1 * (z1 - x10)     # 估计当前状态
        p11 = (1 - K1) * p10

        x21 = x11
        p21 = p11

        low = norm.ppf((1 - confidence)/2, loc=x11, scale=np.sqrt(p11))
        high = norm.ppf(confidence + (1 - confidence)/2, loc=x11, scale=np.sqrt(p11))
        low_bound.append(low)
        high_bound.append(high)
        kalman_gain.append(K1)
        actual.append(height_true)
        measurements.append(z1)
        estimate.append(x11)
        prediction.append(x10)

        x00 = x21
        p00 = p21

    plt.plot(kalman_gain, label="Kalman Gain")
    plt.legend()
    plt.show()

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    plt.plot(prediction, label="prediction")
    x = np.linspace(0, len(actual) - 1, num=len(actual))
    plt.fill_between(x, low_bound, high_bound, alpha=0.5,
                     label='confidence interval')
    plt.legend()
    plt.show()


def main6():
    """估计缸中液体的温度"""
    noise_var = 0.0001                      # 假设系统模型准确，过程噪声的方差
    tool_error = 0.1                          # 温度计误差（标准差）是5米
    temperature_estimate = 60               # 估计温度
    std_estimate = 100                      # 温度计估计标准差
    var_estimate = std_estimate ** 2        # 肉眼估计方差

    x00 = temperature_estimate
    p00 = var_estimate + noise_var

    confidence = 0.95

    kalman_gain = []
    actual = []
    measurements = []
    estimate = []
    prediction = []
    low_bound = []
    high_bound = []

    real = [50.005, 49.994, 49.993, 50.001, 50.006, 49.998, 50.021, 50.005, 50, 49.997]    # 每次测量时真实的液体温度
    z1s = [49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965, 50.114]   # 温度的测量值

    for i in range(len(z1s)):
        x10 = x00
        p10 = p00

        z1 = z1s[i]
        # z1 = height_true + np.random.randint(-tool_error, tool_error)
        r1 = tool_error ** 2

        K1 = p10 / (p10+r1)             # 卡尔曼增益
        x11 = x10 + K1 * (z1 - x10)     # 估计当前状态
        p11 = (1 - K1) * p10

        x21 = x11
        p21 = p11 + noise_var

        low = norm.ppf((1 - confidence)/2, loc=x11, scale=np.sqrt(p11))
        high = norm.ppf(confidence + (1 - confidence)/2, loc=x11, scale=np.sqrt(p11))
        low_bound.append(low)
        high_bound.append(high)
        kalman_gain.append(K1)
        actual.append(real[i])
        measurements.append(z1)
        estimate.append(x11)
        prediction.append(x10)

        x00 = x21
        p00 = p21

    plt.plot(kalman_gain, label="Kalman Gain")
    plt.legend()
    plt.show()

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    # plt.plot(prediction, label="prediction")
    x = np.linspace(0, len(actual) - 1, num=len(actual))
    plt.fill_between(x, low_bound, high_bound, alpha=0.5,
                     label='confidence interval')
    plt.legend()
    plt.show()


def main7():
    """估计缸中液体的温度"""
    noise_var = 0.001                         # 假设系统模型准确，过程噪声的方差
    tool_error = 0.1                        # 温度计误差（标准差）是5米
    temperature_estimate = 10               # 估计温度
    std_estimate = 100                      # 温度计估计标准差
    var_estimate = std_estimate ** 2        # 肉眼估计方差


    x00 = temperature_estimate
    p00 = var_estimate + noise_var   # p00 随着迭代不断增加noise_var，导致卡尔曼增益月来越接近1,就相当于越来越详细测量值，测量值有噪音

    confidence = 0.95

    kalman_gain = []
    actual = []
    measurements = []
    estimate = []
    prediction = []
    low_bound = []
    high_bound = []

    real = [50.505, 50.994, 51.493, 52.001, 52.506, 52.998, 53.521, 54.005, 54.5, 54.997]    # 每次测量时真实的液体温度
    z1s = [50.486, 50.963, 51.597, 52.001, 52.518, 53.05, 53.438, 53.858, 54.465, 55.114]   # 温度的测量值

    for i in range(len(z1s)):
        x10 = x00
        p10 = p00

        z1 = z1s[i]
        # z1 = height_true + np.random.randint(-tool_error, tool_error)
        r1 = tool_error ** 2

        K1 = p10 / (p10+r1)             # 卡尔曼增益
        x11 = x10 + K1 * (z1 - x10)     # 估计当前状态
        p11 = (1 - K1) * p10

        x21 = x11 + 0.9   # 调整
        p21 = p11 + noise_var

        low = norm.ppf((1 - confidence)/2, loc=x11, scale=np.sqrt(p11))
        high = norm.ppf(confidence + (1 - confidence)/2, loc=x11, scale=np.sqrt(p11))
        low_bound.append(low)
        high_bound.append(high)
        kalman_gain.append(K1)
        actual.append(real[i])
        measurements.append(z1)
        estimate.append(x11)
        prediction.append(x10)

        x00 = x21
        p00 = p21

    plt.plot(kalman_gain, label="Kalman Gain")
    plt.legend()
    plt.show()

    plt.plot(actual, label="actual")
    plt.plot(measurements, label="measurements")
    plt.plot(estimate, label="estimate")
    # plt.plot(prediction, label="prediction")
    x = np.linspace(0, len(actual) - 1, num=len(actual))
    plt.fill_between(x, low_bound, high_bound, alpha=0.5,
                     label='confidence interval')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # main()  # 估计黄金重量
    # main2()  # 跟踪直线匀速运动的飞行器
    # main3()  # 跟踪直线加速运动的飞行器
    # main4()  # 用 α−β−γ滤波器跟踪直线加速运动的飞行器
    # main5()  # 估计大楼高度
    # main6()  # 估计缸中液体的温度
    main7()  # 估计加热中的液体的温度 Ⅰ- Ⅱ
