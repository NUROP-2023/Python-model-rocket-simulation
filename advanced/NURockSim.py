"""
    @title 2次元2自由度のモデルロケットフライトシミュレーション
    @author NUROP 内藤正樹
    @date 2023/08/23
"""
import time

import input_output as io
import rocket_simulation as rocksim


def main() -> None:
    INPUT_FILENAME = "いにしゃんず種コン2023"
    THRUST_CUVE_NAME = "C11-3"
    OUTPUT_FILENAME = "いにしゃんず種コン2023シミュレーション結果"
    CSV_NAME = OUTPUT_FILENAME

    start = time.time()

    input_file = io.FileInput()
    rocket_params = input_file.load_rocket_params(INPUT_FILENAME)
    thrust_data = input_file.load_thrust_curve(THRUST_CUVE_NAME)
    rocket_simulation = rocksim.RocketSimulation(rocket_params, thrust_data)

    time_array, sol_array = rocket_simulation.simulation()

    max_altitude = max(sol_array[2, :])
    max_velocity = max(sol_array[3, :])
    print(f"max altitude: {max_altitude:.2f} [m]")
    print(f"max velocity: {max_velocity:.2f} [m/s]")

    output_file = io.FileOutput()

    end = time.time()

    print(f"processing time: {(end - start) * 1000:.2f} [ms]")
    output_file.plot_graph(time_array, sol_array)
    output_file.save_graph(OUTPUT_FILENAME)
    output_file.save_to_csv(CSV_NAME, time_array, sol_array)


if __name__ == "__main__":
    main()
