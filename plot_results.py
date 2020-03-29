import csv
import matplotlib.font_manager as fnt
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    thresholds = [384, 448, 480, 492]
    fitnessfs = ["f1", "f2", "f3"]
    colors = [["#003333", "#330033", "#333300"],
              ["#006666", "#660066", "#666600"],
              ["#009999", "#990099", "#999900"],
              ["#00cccc", "#cc00cc", "#cccc00"],
              ["#00ffff", "#ff00ff", "#ffff00"],]
    thresh_epoch = {}
    for t in thresholds:
        lists = {}
        for f in fitnessfs:
            lists[f] = list()
        thresh_epoch[t] = lists

    for f in fitnessfs:
        fdir = os.path.join("output", f)
        for p in os.listdir(fdir):
            if ".csv" not in p:
                continue
            with open(os.path.join(fdir, p)) as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader) # skip header
                tti = 0
                for row in csv_reader:
                    if int(row[3]) >= thresholds[tti]:
                        thresh_epoch[thresholds[tti]][f].append(int(row[0]))
                        tti += 1
                        if tti >= len(thresholds):
                            break
                if tti < len(thresholds):
                    for i in range(tti, len(thresholds)):
                        thresh_epoch[thresholds[i]][f].append(1000000)

    figure, axes = plt.subplots(1, len(thresholds), sharey = True)
    figure.suptitle("Comparison of effectivity to reach non-linearity values")
    axes[0].set_ylabel("Number of epochs", fontweight="bold", fontsize="large")
    for i, a in enumerate(axes):
        a.set_title("Nf ≥ {}".format(thresholds[i]), fontweight="bold")
        for j, f in enumerate(fitnessfs):
            a.boxplot(thresh_epoch[thresholds[i]][f], labels = [f], patch_artist = True,
                      boxprops = {"facecolor": colors[len(thresh_epoch[thresholds[i]][f]) // 20][j]},
                      positions = [j], medianprops = {"linewidth": 2.5}, widths = [0.5])

    figure.savefig("comparison.svg", bbox_inches="tight")

if __name__ == "__main__":
    main()
