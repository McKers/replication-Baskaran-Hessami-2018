""" This module contains functions for plotting the graphs presented in the replication notebook"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

from localreg import *
from auxiliary.data_processing import *


def bin_fct(data, sbins):
    """ Creates column which indicates the bin of each observation according to the "margin_1" variable

    :param (df)         data: df that contains "margin_1"
    :param (int)        sbins: length of bin
    :return(df):        df with bin indication as variable
    """
    data.loc[:, "bin"] = data["margin_1"] - np.mod(data["margin_1"], sbins) + sbins / 2
    return data


def smooth(x, y, xgrid):
    """ Approximates the linear relationship between x and y by a random subsample

    :param (np.array)   x: x-axis data points
    :param np.array     y: y-axis data points
    :param np.array     xgrid: interpolation points
    :return np.array: approximate/smoothed y-values
    """
    samples = np.random.choice(len(x), len(x), replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = localreg(x_s, y_s, x0=None, degree=1, kernel=triangular, width=19.08094)
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)

    return y_grid


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height. Intended to be called by (e.g)
        figure1_plot().

    :param rects:       matplotlib ax.bar
    :param ax:          matplotlib figure (ax)
    :return:            counts displayed on bar
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def data_fig1_a(data, ext=False):
    """ Process data for plotting figure 1a.

    :param (df)         data: main dataframe of analysis
    :param (bool)       ext: if True, returns data for extension 6.1
    :return:            list(s) of int
    """
    rslt_1 = list(data[['gkz', 'jahr']].drop_duplicates()["jahr"].value_counts().sort_values())

    if ext:
        data_rdd = get_rdd_data(data, female=False)
        rslt_2 = list(data_rdd[['gkz', 'jahr']].drop_duplicates()["jahr"].value_counts().sort_values())

        return rslt_1, rslt_2

    return rslt_1


def data_fig1_b(data, ext=False):
    """Process data for plotting figure 1b.

    :param (df)         data: main dataframe of analysis
    :param (bool)       ext: if True, returns data for extension 6.1
    :return:            list(s) of int
    """
    data_0 = list(data.groupby(data["jahr"]).size())

    if ext:
        temp_1 = data.loc[data["female"] == 1]
        data_1 = list(temp_1.groupby(temp_1["jahr"]).size())

        temp_2 = get_rdd_data(data, female=False)
        data_2 = list(temp_2.groupby(temp_2["jahr"]).size())

        temp_3 = get_rdd_data(data, female=True)
        data_3 = list(temp_3.groupby(temp_3["jahr"]).size())

        return data_0, data_1, data_2, data_3
    return data_0


def figure1_plot(data1, data2):
    """Plots the histogram for figure 1.

    :param              data1: list obtained by data_fig1_a()
    :param              data2: list obtained by data_fig1_b()
    :return:            histogram/figure1
    """
    labels = [2001, 2006, 2011, 2016]

    x = np.arange(len(labels))  # the label locations

    width = 0.5  # the width of the bars of first  graph

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)

    bar1a = ax0.bar(x, data1, width)
    bar1b = ax1.bar(x, data2, width)

    #  label subplot 1 & 2
    autolabel(bar1a, ax0)
    autolabel(bar1b, ax1)

    # specs subplot 1
    ax0.set_title('Number of municipalities')
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.grid(True, axis='y')
    fig.tight_layout()

    # specs subplot 2
    ax1.set_title('Number of Council Candidates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.grid(True, axis='y')
    fig.tight_layout()

    plt.show()


def figure1_plot_extension(data1, data2, data3, data4, data5, data6):
    """Plots the histogram for the extension in section 6.1.

    :param (tuple)      data1: (list[0] obtained by data_fig1_a(),"label")
    :param (tuple)      data2: (list[1] obtained by data_fig1_a(),"label")
    :param (tuple)      data3: (list[0] obtained by data_fig1_b(),"label")
    :param (tuple)      data4: (list[1] obtained by data_fig1_b(),"label")
    :param (tuple)      data5: (list[2] obtained by data_fig1_b(),"label")
    :param (tuple)      data6: (list[3] obtained by data_fig1_b(),"label")
    :return:            histogram/figure  in section 6.1
    """
    labels = [2001, 2006, 2011, 2016]

    x = np.arange(len(labels))  # the label locations

    width_a = 0.40  # the width of the bars of first  graph
    width_b = 0.30  # the width of the bars of second  graph

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
    bar1a = ax0.bar(x - width_a / 2, data1[0], width_a, label=data1[1])
    bar2a = ax0.bar(x + width_a / 2, data2[0], width_a, label=data2[1])

    bar1b = ax1.bar(x - width_b, data3[0], width_b / 2, label=data3[1])
    bar2b = ax1.bar(x - width_b / 3, data4[0], width_b / 2, label=data4[1])
    bar3b = ax1.bar(x + width_b / 3, data5[0], width_b / 2, label=data5[1])
    bar4b = ax1.bar(x + width_b, data6[0], width_b / 2, label=data6[1])

    #  label subplot 1
    autolabel(bar1a, ax0)
    autolabel(bar2a, ax0)

    #  label subplot 2
    autolabel(bar1b, ax1)
    autolabel(bar2b, ax1)
    autolabel(bar3b, ax1)
    autolabel(bar4b, ax1)

    # specs subplot 1
    ax0.set_title('Number of municipalities')
    ax0.legend()
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.grid(True, axis='y')
    fig.tight_layout()

    # specs subplot 2
    ax1.set_title('Number of Council Candidates')
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.grid(True, axis='y')
    fig.tight_layout()

    plt.show()


def rdd_plot(data, sbins, bw, k, calc_points, dependant_var):
    """Plots smoothed local regression with bootstrapped CIs on both sides of "margin_1".

    :param (df)         data: df that contains "margin_1" and parameter dependant_var
    :param (int)        sbins: length of bin
    :param (int)        bw: bandwidth for local regression
    :param (int)        k: iterations of resampling by bootstrapping
    :param (int)        calc_points: points where to calculate smoothed value
    :param (str)        dependant_var: name of dependant variable in df
    :return:            plot
    """
    temp_df = bin_fct(data, sbins)
    avg_rank_impr = temp_df.groupby(temp_df["bin"]).mean()[dependant_var]

    x = range(-30, 30, sbins)
    vic_marg = x - np.mod(x, sbins) + sbins / 2

    df_figure2 = pd.DataFrame([vic_marg, avg_rank_impr], index=["vic_marg", "rank_imp"]).transpose()

    df_neg = temp_df.loc[(temp_df["margin_1"] < 0)].sort_values(by=["margin_1"])
    df_pos = temp_df.loc[(temp_df["margin_1"] > 0)].sort_values(by=["margin_1"])

    y1 = np.asarray(df_neg[dependant_var])
    y2 = np.asarray(df_pos[dependant_var])

    x1 = np.asarray(df_neg["margin_1"])
    x2 = np.asarray(df_pos["margin_1"])

    x_sm1 = x1[0::calc_points]
    x_sm2 = x2[0::calc_points]

    reg_1 = localreg(x1, y1, x0=x_sm1, degree=1, kernel=triangular, width=bw)
    reg_2 = localreg(x2, y2, x0=x_sm2, degree=1, kernel=triangular, width=bw)

    xgrid1 = np.linspace(-30, 0, 50)
    xgrid2 = np.linspace(0, 30, 50)

    smooths1 = np.stack([smooth(x1, y1, xgrid1) for i in range(k)]).T
    smooths2 = np.stack([smooth(x2, y2, xgrid2) for i in range(k)]).T

    mean_neg = np.nanmean(smooths1, axis=1)
    stderr_neg = np.nanstd(smooths1, axis=1, ddof=0)

    mean_pos = np.nanmean(smooths2, axis=1)
    stderr_pos = np.nanstd(smooths2, axis=1, ddof=0)

    fig, (ax0) = plt.subplots(1, 1, figsize=(12, 8), tight_layout=True)

    plt.fill_between(xgrid1, mean_neg - 1.96 * stderr_neg, mean_neg + 1.96 * stderr_neg, alpha=0.25)

    plt.fill_between(xgrid2, mean_pos - 1.96 * stderr_pos, mean_pos + 1.96 * stderr_pos, alpha=0.25)

    plt.axvline(0, linewidth=0.4, color='r')
    ax0.grid(True)
    ax0.scatter(df_figure2["vic_marg"], df_figure2["rank_imp"])
    ax0.plot(x_sm1, reg_1)
    ax0.plot(x_sm2, reg_2)
    plt.xlabel("% Margin of Victory")
    plt.ylabel("Average Rank Improvment")
    ax0.axis([-30, 30, -6, 6])
    plt.show()
