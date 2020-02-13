import numpy as np
import matplotlib.pyplot as plt
from math import exp


def rbfnet(x, y):
    C1 = (0, 0)
    C2 = (1, 1)
    beta1 = 4
    beta2 = 4
    w1 = 1
    w2 = 1

    v1 = - beta1 * ((x - C1[0]) ** 2 + (y - C1[1]) ** 2)
    v2 = - beta2 * ((x - C2[0]) ** 2 + (y - C2[1]) ** 2)

    return w1 * exp(v1) + w2 * exp(v2)


if __name__ == "__main__":
    xv, yv = (-1, 2)
    n = 100
    x = np.linspace(xv, yv, n)
    y = np.linspace(xv, yv, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Z[i, j] = rbfnet(X[i, j], Y[i, j])

    plt.figure(figsize=(5, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    pcm = plt.pcolormesh(X, Y, Z, cmap='RdBu_r')
    plt.colorbar(pcm)
    plt.scatter([0, 1], [0, 1], c='g', marker='^', linewidths=2, label="+")
    plt.scatter([0, 1], [1, 0], c='y', marker='o', linewidths=2, label="-")
    plt.xlim(xv, yv)
    plt.ylim(xv, yv)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.title("用于解决异或问题的rbf网络")
    # plt.show()
    plt.savefig("nn\\rbfnet_xor.png", dpi=800)
