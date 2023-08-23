"""
    @title モデルロケットの1次元質点系シミュレーション
    @author NUROP 内藤正樹
    @date 2023/08/22
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# エンジン諸元
# C11-3の場合
TOTAL_IMPULSE = 8.8         # トータルインパルス [N s]
BURN_TIME = 0.8             # 燃焼時間 [s] 
PROP_MASS = 12 / 1000       # 推進薬量 [kg]
ENGINE_MASS = 35.3 / 1000   # エンジン質量[kg]

# 機体諸元
# いにしゃんず種コン2023の場合
DIAMETER = 56 / 1000        # 機体直径 [m]
BODY_MASS = 124 / 1000      # エンジン抜き機体質量 [kg]
C_D = 0.55                  # 抗力係数 [-]   

# 物理定数
RHO = 1.225                 # 大気密度 [kg/m^3]
GRAVITY = 9.80665           # 重力加速度 [m/s^2]

# 予め計算
AVERAGE_THRUST = TOTAL_IMPULSE / BURN_TIME              # 平均推力 [N]
AREA = math.pi * (DIAMETER)**2 / 4                      # 断面積 [m^2]
AVERAGE_MASS = BODY_MASS + ENGINE_MASS - PROP_MASS / 2  # 打ち上げ平均質量[kg]
W = AVERAGE_MASS * GRAVITY                              # 重力 [N]

# 解析時間
ANALYSIS_TIME = 10              # 解析時間 [s]
DT = 0.01                       # 時間刻み [s]
DIV = int(ANALYSIS_TIME / DT)   # 分割数

# ファイル保存名
FIGURE_NAME = "いにしゃんず種コン2023"


def simulation():
    x_0 = 0
    v_0 = 0
    X_0 = [x_0, v_0]

    t = np.linspace(0, ANALYSIS_TIME, DIV)
    sol = odeint(eom, X_0, t)

    return t, sol


def eom(X, t):
    x, v = X

    thrust = 0
    if t <= BURN_TIME:
        thrust = AVERAGE_THRUST

    drag = (RHO * v * abs(v) * AREA * C_D) / 2

    a = (thrust - drag - W) / AVERAGE_MASS

    return [v, a]


def plot_graph(t, x, v):
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot()

    ax.plot(t, x, label = r"$x$")
    ax.plot(t, v, label = r"$v$")

    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$x$ [m], $v$ [m/s]')
    ax.grid(color='black',linestyle='dotted')
    ax.legend()

    plt.show()

    return fig


def save_fig(fig):
    fig.savefig(f"{FIGURE_NAME}.png", dpi=300)


if __name__ == '__main__':
    t, sol = simulation()

    x_sol = sol[:, 0]
    v_sol = sol[:, 1]

    max_altitude = max(x_sol)
    max_velocity = max(v_sol)
    print(f"max altitude: {max_altitude:.1f} [m]")
    print(f"max velocity: {max_velocity:.1f} [m/s]")

    fig = plot_graph(t, x_sol, v_sol)
    save_fig(fig)