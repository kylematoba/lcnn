import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 12})


def plot(csv_list, out_file, legends, xlabel, ylabel):
    """
    csv_list: list of csv files to plot, one per legend label
    legends: tuple of legends of csv files in csv_list
    plotval: "loss" or "accuracy"        
    """

    i = 0
    for c in csv_list:
        df = pd.read_csv(c, header=None).T
        plt.plot(df[0].values, df[1].values, linewidth=4)
        i+=1

    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim((0., 1.0))
    plt.legend(legends, frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()
