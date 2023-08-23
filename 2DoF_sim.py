"""
    @title 2次元2自由度のモデルロケットフライトシミュレーション
    @author NUROP 内藤正樹
    @date 2023/08/23
"""


import yaml                         # 設定ファイル読み込み
import math                         # 円周率
import matplotlib.pyplot as plt     # グラフ描画
from scipy.integrate import odeint  # 常微分方程式を解く
import numpy as np                  # 数値計算ライブラリ
import csv                          # csv出力


class RocketSimulation:
    def __init__(self, rocket_params):
        # 物理定数
        self.RHO = 1.225                 # 大気密度 [kg/m^3]
        self.GRAVITY = 9.80665           # 重力加速度 [m/s^2]

        total_impulse = rocket_params["トータルインパルス [N s]"]
        propellant_mass = rocket_params["推進薬量 [g]"] / 1000
        engine_mass = rocket_params["エンジン質量 [g]"]  / 1000
        diameter = rocket_params["機体直径 [mm]"] / 1000
        body_mass = rocket_params["エンジン抜き機体質量 [g]"] / 1000

        self.burn_time = rocket_params["燃焼時間 [s]"]
        self.C_Dx = rocket_params["x軸方向抗力係数 [-]"]
        self.C_Dy = rocket_params["y軸方向抗力係数 [-]"]

        self.launch_angle = np.radians(rocket_params["射角 [deg]"])
        self.wind_speed = rocket_params["平均風速 [m/s]"]

        self.analysis_time = rocket_params["解析時間 [s]"]
        self.time_step = rocket_params["時間刻み [s]"]

        # 予め計算
        self.average_thrust = total_impulse / self.burn_time            # 平均推力 [N]
        self.area = math.pi * diameter**2 / 4                           # 断面積 [m^2]
        self.mean_mass = body_mass + engine_mass - propellant_mass / 2  # 打ち上げ平均質量[kg]
        self.mean_weight = self.mean_mass * self.GRAVITY    # 重力 [N]

    def simulation(self):
        """
        odeソルバーを用いて微分方程式を解く

        戻り値
            t_eval      時間の配列
            sol[:, 0]   x軸方向位置
            sol[:, 1]   x軸方向速度
            sol[:, 2]   y軸方向位置
            sol[:, 3]   y軸方向速度
        """

        X = [0, 0, 0, 0]   # [x0, v_x0, y0, v_y0]

        t_eval = np.arange(0, self.analysis_time, self.time_step)
        t_eval = np.append(t_eval, self.analysis_time)
        sol = odeint(self.__eom, X, t_eval)

        return t_eval, sol
    
    def __eom (self, X, t):
        """
        運動方程式 Equation of Motion
        
        引数
            X  ベクトル [x, v_x, y, v_y]
            t  スカラ
        戻り値
            [dv_x, da_x, dv_y, da_y]
        """

        x, v_x, y, v_y = X

        thrst = 0
        if t <= self.burn_time:
            thrst = self.average_thrust

        thrst_x = thrst * np.sin(self.launch_angle)
        thrst_y = thrst * np.cos(self.launch_angle)

        air_speed = v_x + self.wind_speed
        drag_x = (self.RHO * air_speed * abs(air_speed) * self.area * self.C_Dx) / 2
        drag_y = (self.RHO * v_y * abs(v_y) * self.area * self.C_Dy) / 2

        weight = self.mean_mass * self.GRAVITY

        a_x = (thrst_x - drag_x) / self.mean_mass
        a_y = (thrst_y - drag_y - weight) / self.mean_mass

        return [v_x, a_x, v_y, a_y]

    def plot_graph(self, t, sol):
        self.fig = plt.figure(tight_layout = True)

        ax11 = self.fig.add_subplot(2, 1, 1)
        ax11.plot(t, sol[:, 0], "C0", label = r"$x$")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$x$ [m]")
        ax11.grid(color = "black", linestyle = "dotted")

        ax12 = ax11.twinx()
        ax12.set_ylabel(r"$\dot{x}$ [m/s]")
        ax12.plot(t, sol[:, 1], "C1", label = r"$\dot{x}$")

        h1, l1 = ax11.get_legend_handles_labels()
        h2, l2 = ax12.get_legend_handles_labels()
        ax11.legend(h1+h2, l1+l2)

        ax21 = self.fig.add_subplot(2, 1, 2)
        ax21.plot(t, sol[:, 2], "C0", label = r"$y$")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$y$ [m]")
        ax21.legend()
        ax21.grid(color = "black", linestyle = "dotted")

        ax22 = ax21.twinx()
        ax22.set_ylabel(r"$\dot{y}$ [m/s]")
        ax22.plot(t, sol[:, 3], "C1", label = r"$\dot{y}$")

        h1, l1 = ax21.get_legend_handles_labels()
        h2, l2 = ax22.get_legend_handles_labels()
        ax22.legend(h1+h2, l1+l2)

        plt.show()

    def save_graph(self, fig_name):
        self.fig.savefig(f"{fig_name}.png", dpi = 300, transparent = True)

    def save_to_csv(self, csv_name, t, sol):
        t = t.reshape(len(t), 1)
        data = np.hstack([t, sol])
        with open(f"{csv_name}.csv", "w") as file:
            writer = csv.writer(file, lineterminator = "\n")
            writer.writerow(["t","x","v_x","y","y_v"])
            writer.writerows(data)


def load_rocket_params(file_name):
        with open(f"{file_name}.yaml", encoding = "utf-8") as f:
            rocket_params = yaml.load(f, Loader = yaml.Loader)
        
        return rocket_params

def main():
    INPUT_FILENAME = "setting_file"
    OUTPUT_FILENAME = "simulation_result"
    CSV_NAME = OUTPUT_FILENAME

    rocket_params = load_rocket_params(INPUT_FILENAME)
    rocket_simulation = RocketSimulation(rocket_params)

    time_array, sol_array = rocket_simulation.simulation()

    max_altitude = max(sol_array[:, 2])
    max_velocity = max(sol_array[:, 3])
    print(f"max altitude: {max_altitude:.2f} [m]")
    print(f"max velocity: {max_velocity:.2f} [m/s]")

    rocket_simulation.plot_graph(time_array, sol_array)
    rocket_simulation.save_graph(OUTPUT_FILENAME)
    rocket_simulation.save_to_csv(CSV_NAME, time_array, sol_array)


if __name__ == "__main__":
    main()