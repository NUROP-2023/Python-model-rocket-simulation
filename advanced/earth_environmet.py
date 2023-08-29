class EarthEnvironment:
    def __init__(self) -> None:
        pass

    def air_density(self, alt: float) -> float:
        T_0 = 288.15        # 海面気温 [K]
        P_0 = 101325        # 海面気圧 [Pa]

        temperature = T_0 - 0.0065 * alt
        pressure = P_0 * (T_0 / temperature) ** (-5.256)
        rho = 0.0034837 * pressure / temperature

        return rho

    def gravity_constant(self) -> float:
        GRAVITY = 9.80665           # 重力加速度 [m/s^2]

        return GRAVITY

    def wind_speed(self, alt: float, ground_wind_speed: float) -> float:
        ALPHA = 0.3     # 風のべき乗則における係数（都市部）
        Z_0 = 10        # 地上風計測高度[m]

        if alt >= 0:
            wind_speed = ground_wind_speed * (alt / Z_0) ** ALPHA
        else:
            wind_speed = ground_wind_speed

        return wind_speed


if __name__ == "__main__":
    earh_env = EarthEnvironment()

    alt = 100
    ground_ws = 5

    print(f"大気密度: {earh_env.air_density(alt)} [kg/m^3]")
    print(f"重力加速度: {earh_env.gravity_constant()} [m/s^2]")
    print(f"風速: {earh_env.wind_speed(alt, ground_ws)} [m/s]")
