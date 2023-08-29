import math
from scipy.integrate import solve_ivp
import numpy as np
from scipy import interpolate

import earth_environmet as env


class RocketSimulation:
    def __init__(self, rocket_params: dict, thrust_data: dict) -> None:
        # エンジン関係
        self.total_impulse = rocket_params["トータルインパルス [N s]"]
        self.propellant_mass = rocket_params["推進薬量 [g]"] / 1000
        self.engine_mass = rocket_params["エンジン質量 [g]"] / 1000
        self.burn_time = rocket_params["燃焼時間 [s]"]

        # 機体関係
        self.diameter = rocket_params["機体直径 [mm]"] / 1000
        self.body_mass = rocket_params["エンジン抜き機体質量 [g]"] / 1000
        self.C_Dx = rocket_params["x軸方向抗力係数 [-]"]
        self.C_Dy = rocket_params["y軸方向抗力係数 [-]"]

        # パラシュート関係
        self.delay_time = rocket_params["延時時間 [s]"]
        self.parachute_diameter = rocket_params["パラシュート直径 [mm]"] / 1000
        self.parachute_C_D = rocket_params["パラシュート抗力係数 [-]"]

        # 打ち上げ条件
        self.launch_angle = np.radians(rocket_params["射角 [deg]"])
        self.ground_wind_speed = rocket_params["平均風速 [m/s]"]

        # 解析条件
        self.analysis_time = rocket_params["解析時間 [s]"]
        self.time_step = rocket_params["時間刻み [s]"]

        # 地球環境計算
        self.earh_env = env.EarthEnvironment()
        self.gravity_constant = self.earh_env.gravity_constant()

        # 推力
        time = thrust_data["time"]
        thrust = thrust_data["thrust"]
        self.thrst_curve = interpolate.interp1d(time, thrust, kind="linear")

        # 予め計算
        # 平均推力 [N]
        self.average_thrust = self.total_impulse / self.burn_time
        # 断面積 [m^2]
        self.area = math.pi * self.diameter**2 / 4
        # パラシュート断面積 [m^2]
        self.parachute_area = math.pi * self.parachute_diameter**2 / 4

    def simulation(self) -> tuple[np.ndarray, np.ndarray]:
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

        sol = solve_ivp(self.__eom, [0, self.analysis_time], X,
                        max_step=0.01, events=self.hit_ground)

        return sol.t, sol.y

    def hit_ground(self, t: float, X: list):
        return X[2] + 0.01

    hit_ground.terminal = True

    def __eom(self, t: float, X: list) -> list:
        """
        運動方程式 Equation of Motion

        引数
            X  ベクトル [x, v_x, y, v_y]
            t  スカラ
        戻り値
            [dv_x, da_x, dv_y, da_y]
        """

        x, v_x, y, v_y = X

        thrust_x, thrust_y = self.__thrust(t)
        drag_x, drag_y = self.__drag(t, X)

        mass = self.__mass(t)
        weight = self.__weight(mass)

        a_x = (thrust_x - drag_x) / mass
        a_y = (thrust_y - drag_y - weight) / mass

        return [v_x, a_x, v_y, a_y]

    def __drag(self, t: float, X: list) -> tuple[float, float]:
        x, v_x, y, v_y = X

        rho = self.earh_env.air_density(y)
        air_speed = v_x + self.earh_env.wind_speed(y, self.ground_wind_speed)

        drag_x = (rho * air_speed * abs(air_speed) * self.area * self.C_Dx) / 2

        if t <= self.delay_time:
            drag_y = (rho * v_y * abs(v_y) * self.area * self.C_Dy) / 2
        else:
            drag_y = (rho * v_y * abs(v_y) * self.parachute_area
                      * self.parachute_C_D) / 2

        return drag_x, drag_y

    def __mass(self, t: float) -> float:
        mass = self.body_mass + self.engine_mass

        if t <= self.burn_time:
            mass = self.body_mass + self.engine_mass
            - self.propellant_mass / self.burn_time * t

        return mass

    def __weight(self, mass: float) -> tuple[float, float]:
        return mass * self.gravity_constant

    def __thrust(self, t: float) -> tuple[float, float]:
        thrst = 0
        if t <= self.burn_time:
            thrst = self.thrst_curve(t)

        thrst_x = thrst * np.sin(self.launch_angle)
        thrst_y = thrst * np.cos(self.launch_angle)

        return thrst_x, thrst_y


if __name__ == "__main__":
    rocket_params = {
        "トータルインパルス [N s]": 8.8,
        "燃焼時間 [s]": 0.8,
        "推進薬量 [g]": 12,
        "エンジン質量 [g]": 35.3,
        "機体直径 [mm]": 56,
        "エンジン抜き機体質量 [g]": 124,
        "x軸方向抗力係数 [-]": 22,
        "y軸方向抗力係数 [-]": 0.55,
        "射角 [deg]": 0,
        "平均風速 [m/s]": 0,
        "解析時間 [s]": 10,
        "時間刻み [s]": 0.01,
        "延時時間 [s]": 3,
        "パラシュート直径 [mm]": 450,
        "パラシュート抗力係数 [-]": 0.75
    }

    import pandas as pd
    thrust_data = pd.read_csv("input/C11-3.csv", encoding="utf-8")

    rocket_simulation = RocketSimulation(rocket_params, thrust_data)
    time_array, sol_array = rocket_simulation.simulation()

    max_altitude = max(sol_array[2, :])
    max_velocity = max(sol_array[3, :])
    print(f"max altitude: {max_altitude:.2f} [m]")
    print(f"max velocity: {max_velocity:.2f} [m/s]")
