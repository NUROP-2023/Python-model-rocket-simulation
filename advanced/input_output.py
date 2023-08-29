import yaml
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"


class FileInput:
    def __init__(self) -> None:
        pass

    def load_rocket_params(self, file_name: str) -> dict:
        with open(f"{INPUT_FOLDER}/{file_name}.yaml", encoding="utf-8") as f:
            rocket_params = yaml.load(f, Loader=yaml.Loader)

        return rocket_params

    def load_thrust_curve(self, file_name: str) -> dict:
        thrust_data = pd.read_csv(f"{INPUT_FOLDER}/{file_name}.csv",
                                  encoding="utf-8")

        return thrust_data


class FileOutput:
    def __init__(self) -> None:
        pass

    def plot_graph(self, t: np.ndarray, sol: np.ndarray) -> None:
        self.fig = plt.figure(tight_layout=True)

        ax11 = self.fig.add_subplot(2, 1, 1)
        ax11.plot(t, sol[0, :], "C0", label=r"$x$")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$x$ [m]")
        ax11.grid(color="black", linestyle="dotted")

        ax12 = ax11.twinx()
        ax12.set_ylabel(r"$\dot{x}$ [m/s]")
        ax12.plot(t, sol[1, :], "C1", label=r"$\dot{x}$")

        h1, l1 = ax11.get_legend_handles_labels()
        h2, l2 = ax12.get_legend_handles_labels()
        ax11.legend(h1+h2, l1+l2)

        ax21 = self.fig.add_subplot(2, 1, 2)
        ax21.plot(t, sol[2, :], "C0", label=r"$y$")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$y$ [m]")
        ax21.legend()
        ax21.grid(color="black", linestyle="dotted")

        ax22 = ax21.twinx()
        ax22.set_ylabel(r"$\dot{y}$ [m/s]")
        ax22.plot(t, sol[3, :], "C1", label=r"$\dot{y}$")

        h1, l1 = ax21.get_legend_handles_labels()
        h2, l2 = ax22.get_legend_handles_labels()
        ax22.legend(h1+h2, l1+l2)

        plt.show()

    def save_graph(self, fig_name: str) -> None:
        os.makedirs(f"{OUTPUT_FOLDER}", exist_ok=True)
        self.fig.savefig(f"{OUTPUT_FOLDER}/{fig_name}.png",
                         dpi=300, transparent=True)

    def save_to_csv(self, csv_name: str,
                    t: np.ndarray,
                    sol: np.ndarray) -> None:

        os.makedirs(f"{OUTPUT_FOLDER}", exist_ok=True)
        t = t.reshape(1, len(t))
        data = np.vstack([t, sol])

        with open(f"{OUTPUT_FOLDER}/{csv_name}.csv", "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["t", "x", "v_x", "y", "y_v"])
            writer.writerows(data.T)
