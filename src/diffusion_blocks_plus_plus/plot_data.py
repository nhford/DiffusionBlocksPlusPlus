import os

import pandas
from matplotlib import pyplot


def generate_plot(name: str):
    data = pandas.read_csv(f"data/{name}.csv")
    os.makedirs("./figures", exist_ok=True)
    pyplot.clf()
    pyplot.title("Loss Over Time")
    pyplot.xlabel("Batch Index")
    pyplot.ylabel("Loss")
    pyplot.plot(data["batch_idx"], data["loss"])
    pyplot.ylim(0, 1)
    pyplot.savefig(f"./figures/{name}.pdf")
