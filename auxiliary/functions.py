import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm_api
import statsmodels as sm
import scipy.interpolate
import warnings

warnings.filterwarnings("ignore")

from localreg import *
from IPython.display import display_html
from scipy import stats


def get_rdd_data(data, female):
    # creates sample for RDD
    if female == 1:

        data1 = data.loc[(data["rdd_sample"] == 1) & (data["female"] == 1)].copy()
    elif female == 0:

        data1 = data.loc[(data["rdd_sample"] == 1)].copy()

    return data1


def ind_fct(vector):
    indic = [1 if abs(x) <= 1 else 0 for x in vector]

    return indic


def data_fig1_a(data):
    temp_1 = data.groupby(data["gkz_jahr"])["jahr"].describe()
    fin_1 = temp_1.groupby(temp_1["mean"]).count()["count"]

    data_rdd = get_rdd_data(data, female=0)
    temp_2 = data_rdd.groupby(data_rdd["gkz_jahr"])["jahr"].describe()
    fin_2 = temp_2.groupby(temp_2["mean"]).count()["count"]

    return [fin_1, fin_2]


def data_fig1_b(data):
    data_0 = data.groupby(data["jahr"]).size()

    temp_1 = data.loc[data["female"] == 1]
    data_1 = temp_1.groupby(temp_1["jahr"]).size()

    temp_2 = get_rdd_data(data, female=0)
    data_2 = temp_2.groupby(temp_2["jahr"]).size()

    temp_3 = get_rdd_data(data, female=1)
    data_3 = temp_3.groupby(temp_3["jahr"]).size()

    return [data_0, data_1, data_2, data_3]


def figure1_plot_a(data1, data2):
    labels = [2001, 2006, 2011, 2016]

    x = np.arange(len(labels))  # the label locations
    width = 0.40  # the width of the bars

    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width / 2, data1[0], width, label=data1[1])
    bar2 = ax.bar(x + width / 2, data2[0], width, label=data2[1])

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bar1)
    autolabel(bar2)

    ax.set_title('Number of municipalities')

    ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.grid(True, axis='y')
    fig.tight_layout()
    plt.show()


def figure1_plot_b(data1, data2, data3, data4):
    labels = [2001, 2006, 2011, 2016]

    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width, data1[0], width / 2, label=data1[1])
    bar2 = ax.bar(x - width / 3, data2[0], width / 2, label=data2[1])
    bar3 = ax.bar(x + width / 3, data3[0], width / 2, label=data3[1])
    bar4 = ax.bar(x + width, data4[0], width / 2, label=data4[1])

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bar1)
    autolabel(bar2)
    autolabel(bar3)
    autolabel(bar4)

    ax.set_title('Number of Council Candidates')

    ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.grid(True, axis='y')
    fig.tight_layout()
    plt.show()


def figure1_plot(data1, data3, ):
    labels = [2001, 2006, 2011, 2016]

    x = np.arange(len(labels))  # the label locations

    def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    width = 0.5  # the width of the bars of first  graph

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
    bar1a = ax0.bar(x, data1, width)

    bar1b = ax1.bar(x, data3, width)

    #  label subplot 1
    autolabel(bar1a, ax0)

    #  label subplot 2
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
    labels = [2001, 2006, 2011, 2016]

    x = np.arange(len(labels))  # the label locations

    def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

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


def get_figure2_old(data, sbins, bw, K):
    # input: rdd-data of bandwith

    temp_df = bin_fct(data, sbins)

    avg_rank_impr = temp_df.groupby(temp_df["bin"]).mean()["gewinn_norm"]

    # get scatter

    x = range(-30, 30, sbins)
    vic_marg = x - np.mod(x, sbins) + sbins / 2

    df_figure2 = pd.DataFrame([vic_marg, avg_rank_impr], index=["vic_marg", "rank_imp"]).transpose()

    df_neg = temp_df.loc[(temp_df["margin_1"] < 0)].sort_values(by=["margin_1"])
    df_pos = temp_df.loc[(temp_df["margin_1"] > 0)].sort_values(by=["margin_1"])

    y1 = np.asarray(df_neg["gewinn_norm"])
    y2 = np.asarray(df_pos["gewinn_norm"])

    x1 = np.asarray(df_neg["margin_1"])
    x2 = np.asarray(df_pos["margin_1"])

    x_sm1 = x1[0::100]
    # x_sm1=x1
    x_sm2 = x2[0::100]
    # x_sm2=x2
    reg_1 = localreg(x1, y1, x0=x_sm1, degree=1, kernel=triangular, width=bw)
    reg_2 = localreg(x2, y2, x0=x_sm2, degree=1, kernel=triangular, width=bw)

    # df_res=pd.DataFrame([x1,x2],[y1,y2],[reg_1,reg_2],columns=['margin_1', 'gewinn_norm', 'prediction'])
    #       h:10.55162        b:16.25238
    xgrid1 = np.linspace(-30, 0, 50)
    xgrid2 = np.linspace(0, 30, 50)

    smooths1 = np.stack([smooth(x1, y1, xgrid1) for k in range(K)]).T
    smooths2 = np.stack([smooth(x2, y2, xgrid2) for k in range(K)]).T

    mean_neg = np.nanmean(smooths1, axis=1)
    stderr_neg = scipy.stats.sem(smooths1, axis=1)
    stderr_neg = np.nanstd(smooths1, axis=1, ddof=0)

    mean_pos = np.nanmean(smooths2, axis=1)
    stderr_pos = scipy.stats.sem(smooths2, axis=1)
    stderr_pos = np.nanstd(smooths2, axis=1, ddof=0)

    plt.fill_between(xgrid1, mean_neg - 1.96 * stderr_neg, mean_neg + 1.96 * stderr_neg, alpha=0.25)

    plt.fill_between(xgrid2, mean_pos - 1.96 * stderr_pos, mean_pos + 1.96 * stderr_pos, alpha=0.25)

    plt.axvline(0, linewidth=0.4, color='r')
    plt.grid(True)
    plt.scatter(df_figure2["vic_marg"], df_figure2["rank_imp"])
    plt.plot(x_sm1, reg_1)
    plt.plot(x_sm2, reg_2)
    plt.xlabel("% Margin of Victory")
    plt.ylabel("Average Rank Improvment")
    plt.title("Figure 2. Rank Improvement of Female Candidates")
    plt.axis([-30, 30, -6, 6])
    plt.show()


def get_figure2(data, sbins, bw, K):
    # input: rdd-data of bandwith

    temp_df = bin_fct(data, sbins)

    avg_rank_impr = temp_df.groupby(temp_df["bin"]).mean()["gewinn_norm"]

    # get scatter

    x = range(-30, 30, sbins)
    vic_marg = x - np.mod(x, sbins) + sbins / 2

    df_figure2 = pd.DataFrame([vic_marg, avg_rank_impr], index=["vic_marg", "rank_imp"]).transpose()

    df_neg = temp_df.loc[(temp_df["margin_1"] < 0)].sort_values(by=["margin_1"])
    df_pos = temp_df.loc[(temp_df["margin_1"] > 0)].sort_values(by=["margin_1"])

    y1 = np.asarray(df_neg["gewinn_norm"])
    y2 = np.asarray(df_pos["gewinn_norm"])

    x1 = np.asarray(df_neg["margin_1"])
    x2 = np.asarray(df_pos["margin_1"])

    x_sm1 = x1[0::100]
    # x_sm1=x1
    x_sm2 = x2[0::100]
    # x_sm2=x2
    reg_1 = localreg(x1, y1, x0=x_sm1, degree=1, kernel=triangular, width=bw)
    reg_2 = localreg(x2, y2, x0=x_sm2, degree=1, kernel=triangular, width=bw)

    # df_res=pd.DataFrame([x1,x2],[y1,y2],[reg_1,reg_2],columns=['margin_1', 'gewinn_norm', 'prediction'])
    #       h:10.55162        b:16.25238
    xgrid1 = np.linspace(-30, 0, 50)
    xgrid2 = np.linspace(0, 30, 50)

    smooths1 = np.stack([smooth(x1, y1, xgrid1) for k in range(K)]).T
    smooths2 = np.stack([smooth(x2, y2, xgrid2) for k in range(K)]).T

    mean_neg = np.nanmean(smooths1, axis=1)
    stderr_neg = scipy.stats.sem(smooths1, axis=1)
    stderr_neg = np.nanstd(smooths1, axis=1, ddof=0)

    mean_pos = np.nanmean(smooths2, axis=1)
    stderr_pos = scipy.stats.sem(smooths2, axis=1)
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
    plt.title("Figure 2. Rank Improvement of Female Candidates")
    ax0.axis([-30, 30, -6, 6])
    plt.show()


def rdd_plot(data, sbins, bw, K, calc_points, dependant_var):
    # input: rdd-data of bandwith
    # df final von
    temp_df = bin_fct(data, sbins)

    avg_rank_impr = temp_df.groupby(temp_df["bin"]).mean()[dependant_var]

    # get scatter

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
    # x_sm1=x1
    x_sm2 = x2[0::calc_points]
    # x_sm2=x2
    reg_1 = localreg(x1, y1, x0=x_sm1, degree=1, kernel=triangular, width=bw)
    reg_2 = localreg(x2, y2, x0=x_sm2, degree=1, kernel=triangular, width=bw)

    # df_res=pd.DataFrame([x1,x2],[y1,y2],[reg_1,reg_2],columns=['margin_1', 'gewinn_norm', 'prediction'])
    #       h:10.55162        b:16.25238
    xgrid1 = np.linspace(-30, 0, 50)
    xgrid2 = np.linspace(0, 30, 50)

    smooths1 = np.stack([smooth(x1, y1, xgrid1) for k in range(K)]).T
    smooths2 = np.stack([smooth(x2, y2, xgrid2) for k in range(K)]).T

    mean_neg = np.nanmean(smooths1, axis=1)
    stderr_neg = scipy.stats.sem(smooths1, axis=1)
    stderr_neg = np.nanstd(smooths1, axis=1, ddof=0)

    mean_pos = np.nanmean(smooths2, axis=1)
    stderr_pos = scipy.stats.sem(smooths2, axis=1)
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
    # plt.title("Figure 2. Rank Improvement of Female Candidates")
    ax0.axis([-30, 30, -6, 6])
    plt.show()


def weight(data, bw):
    temp_i = data['margin_1'] / bw
    ind_i = ind_fct(temp_i)
    weight_i = (1 - abs(temp_i)) * ind_i
    data.loc[:, ('weight' + str(bw) + '')] = weight_i


def table1a(data):
    table1_all = data[["gewinn_norm", "listenplatz_norm", "age", 'non_university_phd', 'university', 'phd', 'architect',
                       'businessmanwoman', 'engineer', 'lawyer', 'civil_administration', 'teacher', 'employed',
                       'selfemployed', 'student', 'retired', 'housewifehusband']].rename(
        columns={"gewinn_norm": "Rank improvement (normalized)", "listenplatz_norm": "Initial list rank (normalized)",
                 "age": "Age", 'non_university_phd': "High school", 'university': 'University', 'phd': 'Phd',
                 'architect': 'Architect', 'businessmanwoman': "Businesswoman/-man", 'engineer': "Engineer",
                 'lawyer': "Lawyer", 'civil_administration': "Civil administration", "teacher": "Teacher",
                 'employed': "Employed", 'selfemployed': "Self-employed", "student": "Student", 'retired': "Retired",
                 'housewifehusband': "Housewife/-husband"})
    table1_a = table1_all.describe(percentiles=[]).transpose()[["count", "mean", "std", "min", "max"]]
    return (table1_a)


def table1b(data):
    data = data[data["female"] == 1]
    table1_f = data[["gewinn_norm", "listenplatz_norm", "age", 'non_university_phd', 'university', 'phd', 'architect',
                     'businessmanwoman', 'engineer', 'lawyer', 'civil_administration', 'teacher', 'employed',
                     'selfemployed', 'student', 'retired', 'housewifehusband']].rename(
        columns={"gewinn_norm": "Rank improvement (normalized)", "listenplatz_norm": "Initial list rank (normalized)",
                 "age": "Age", 'non_university_phd': "High school", 'university': 'University', 'phd': 'Phd',
                 'architect': 'Architect', 'businessmanwoman': "Businesswoman/-man", 'engineer': "Engineer",
                 'lawyer': "Lawyer", 'civil_administration': "Civil administration", "teacher": "Teacher",
                 'employed': "Employed", 'selfemployed': "Self-employed", "student": "Student", 'retired': "Retired",
                 'housewifehusband': "Housewife/-husband"})
    table1_b = table1_f.describe(percentiles=[]).transpose()[["count", "mean", "std", "min", "max"]]
    return (table1_b)


def reg_tab_old(*model):
    # pd.set_option('mode.chained_assignment', None)
    table = pd.DataFrame(
        {'Model': [], 'Female Mayor': [], 'Std.err': [], 'Observations': [], 'Elections': [], 'Municipalities': [],
         'Mean': [], 'Std.err (Mean)': []})

    table = table.set_index(['Model'])

    for counter, i in enumerate(model):

        # added margin reduction
        # data = subset_by_margin(i[0], i[2])
        data_i = subset_by_bw(i[0], i[2])
        weight(data_i, i[2])

        # data["weight" + str(bw) + ""]

        y = data_i[i[3]]
        w = data_i["weight" + str(i[2]) + ""]

        if i[1] == 1:

            x = data_i[["female_mayor", "margin_1", "inter_1"]]
        else:

            x = data_i[["female_mayor", "margin_1", "inter_1", "margin_2", "inter_2"]]

        x = sm_api.add_constant(x)
        wls_model = sm_api.WLS(y, x, weights=w)
        results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': data_i["gkz"]})
        output = [results.params[1], results.bse[1], results.nobs, data_i["gkz_jahr"].value_counts().count(),
                  data_i["gkz"].value_counts().count(), y.mean(), np.std(y)]

        table.loc["Model_(" + str(counter + 1) + ")"] = output
    table = table.round(3)
    return table


def prepare_TableA5(data):
    # data input has to be rdd_sample for female candidates

    x = data[['gkz', 'gkz_jahr', 'gewinn_norm', 'margin_1', 'inter_1', 'margin_2', 'inter_2', 'female_mayor',
              'log_bevoelkerung', 'log_flaeche', 'log_debt_pc', 'log_tottaxrev_pc', 'log_gemeinde_beschaef_pc',
              'log_female_sh_gem_besch', 'log_tot_beschaeft_pc', 'log_female_share_totbesch', 'log_prod_share_tot',
              'log_female_share_prod']]

    x_drop = x.dropna()

    Y = x_drop['gewinn_norm']
    X = x_drop[['log_bevoelkerung', 'log_flaeche', 'log_debt_pc', 'log_tottaxrev_pc', 'log_gemeinde_beschaef_pc',
                'log_female_sh_gem_besch', 'log_tot_beschaeft_pc', 'log_female_share_totbesch', 'log_prod_share_tot',
                'log_female_share_prod']]
    X = sm_api.add_constant(X)

    model = sm_api.OLS(Y, X, missing='drop')
    # results = model.fit(cov_type='cluster', cov_kwds={'groups': x["gkz"]})

    results = model.fit()

    y_hat = results.predict()

    x_drop["y_hat"] = y_hat
    x_another = x_drop.drop(columns=["gewinn_norm"])
    df_final = x_another.drop_duplicates()

    return df_final


def reg_tab(*model):
    # pd.set_option('mode.chained_assignment', None)
    table = pd.DataFrame(
        {'Model': [], 'Female Mayor': [], 'Std.err': [], 'Bandwidth type': [], 'Bandwidth size': [], 'Polynominal': [],
         'Observations': [], 'Elections': [], 'Municipalities': [],
         'Mean': [], 'Std.err (Mean)': []})

    table = table.set_index(['Model'])

    for counter, i in enumerate(model):

        # added margin reduction
        # data = subset_by_margin(i[0], i[2])
        data_i = subset_by_bw(i[0], i[2])
        weight(data_i, i[2])

        # data["weight" + str(bw) + ""]

        y = data_i[i[3]]
        w = data_i["weight" + str(i[2]) + ""]

        if i[1] == 1:

            x = data_i[["female_mayor", "margin_1", "inter_1"]]
        else:

            x = data_i[["female_mayor", "margin_1", "inter_1", "margin_2", "inter_2"]]

        x = sm_api.add_constant(x)
        wls_model = sm_api.WLS(y, x, weights=w)
        results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': data_i["gkz"]})

        if results.pvalues["female_mayor"] <= 0.01:
            result_parameter = str(results.params[1].round(3)) + "***"
        elif 0.01 < results.pvalues["female_mayor"] <= 0.05:
            result_parameter = str(results.params[1].round(3)) + "**"
        elif 0.05 < results.pvalues["female_mayor"] <= 0.1:
            result_parameter = str(results.params[1].round(3)) + "*"
        else:
            result_parameter = str(results.params[1].round(3))

        if i[1] == 1:
            polynominal_i = str("Linear")
        else:
            polynominal_i = str("Quadratic")

        bw_size_i = str(round(i[2], 2))
        bw_type_i = str(i[4])

        output = [result_parameter, results.bse[1], bw_type_i, bw_size_i, polynominal_i, results.nobs,
                  data_i["gkz_jahr"].value_counts().count(),
                  data_i["gkz"].value_counts().count(), y.mean().round(2), np.std(y)]

        table.loc["(" + str(counter + 1) + ")"] = output
    table = table.round(3)
    return table


def reg_tab_ext(*model):
    # pd.set_option('mode.chained_assignment', None)
    table = pd.DataFrame(
        {'Model': [], 'Female Mayor': [], 'Std.err_Female Mayor': [], 'University': [], 'Std.err_University': [],
         'PhD': [], 'Std.err_PhD': [], 'Bandwidth type': [], 'Bandwidth size': [], 'Polynominal': [],
         'Observations': [], 'Elections': [], 'Municipalities': [],
         'Mean': [], 'Std.err (Mean)': []})

    table = table.set_index(['Model'])

    for counter, i in enumerate(model):

        # added margin reduction
        # data = subset_by_margin(i[0], i[2])
        data_i = subset_by_bw(i[0], i[2])
        weight(data_i, i[2])

        # data["weight" + str(bw) + ""]

        y = data_i[i[3]]
        w = data_i["weight" + str(i[2]) + ""]

        if i[1] == 1:

            x = data_i[["female_mayor", "margin_1", "inter_1", 'university', 'phd']]
        else:

            x = data_i[["female_mayor", "margin_1", "inter_1", 'university', 'phd', "margin_2", "inter_2"]]

        x = sm_api.add_constant(x)
        wls_model = sm_api.WLS(y, x, missing='drop', weights=w)
        results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': data_i["gkz"]})

        betas = [1, 2, 3]
        cov = ["female_mayor", 'university', 'phd']

        for j in cov:

            if results.pvalues[j] <= 0.01:
                result_parameter_j = str(results.params[(cov.index(j) + 1)].round(3)) + "***"
            elif 0.01 < results.pvalues[j] <= 0.05:
                result_parameter_j = str(results.params[(cov.index(j) + 1)].round(3)) + "**"
            elif 0.05 < results.pvalues[j] <= 0.1:
                result_parameter_j = str(results.params[(cov.index(j) + 1)].round(3)) + "*"
            else:
                result_parameter_j = str(results.params[(cov.index(j) + 1)].round(3))

            betas[cov.index(j)] = result_parameter_j
            # betas[i]=result_parameter_j

        if i[1] == 1:
            polynominal_i = str("Linear")
        else:
            polynominal_i = str("Quadratic")

        bw_size_i = str(round(i[2], 2))
        bw_type_i = str(i[4])

        output = [betas[0], results.bse[1], betas[1], results.bse[4], betas[2], results.bse[5], bw_type_i, bw_size_i,
                  polynominal_i, results.nobs,
                  data_i["gkz_jahr"].value_counts().count(),
                  data_i["gkz"].value_counts().count(), y.mean().round(2), np.std(y)]

        table.loc["(" + str(counter + 1) + ")"] = output
    table = table.round(3)
    return table


def prepare_ext(data):
    data_temp = data.drop(
        columns=["jahr", "rdd_sample", "female", "elected", "gewinn", "gewinn_dummy", "listenplatz_norm", "joint_party",
                 "age", "architect", "businessmanwoman", "engineer", "lawyer",
                 "civil_administration", "teacher", "employed", "selfemployed", "student", "retired",
                 "housewifehusband", "incumbent_council", "wahlbet", "party", 'log_bevoelkerung', 'log_flaeche',
                 'log_debt_pc', 'log_tottaxrev_pc', 'log_gemeinde_beschaef_pc',
                 'log_female_sh_gem_besch', 'log_tot_beschaeft_pc', 'log_female_share_totbesch', 'log_prod_share_tot',
                 'log_female_share_prod', "female_mayor_full_sample",
                 "sum_years_as_mayor", "mayor_age", "mayor_university", "mayor_employment"])
    data_temp = data_temp.dropna()

    return data_temp


def subset_by_margin2(df, margin):
    # data = data[abs(data["margin_1"]) < margin]
    df = df.loc[(df["margin_1"] < abs(margin))]
    return df


def display_side_by_side(*args):
    # credit goes to stackoverflow user: https://stackoverflow.com/users/508907/ntg
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


def subset_by_margin(data, margin):
    data = data[abs(data["margin_1"]) < margin]

    return data


def subset_by_bw(data, bw):
    df = data[data['margin_1'].between(-bw, bw)].copy()
    return df


def bin_fct(data, Sbins):
    data.loc[:, "bin"] = data["margin_1"] - np.mod(data["margin_1"], Sbins) + Sbins / 2
    return data


def smooth(x, y, xgrid):
    samples = np.random.choice(len(x), len(x), replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = localreg(x_s, y_s, x0=None, degree=1, kernel=triangular, width=19.08094)
    # regularly sample it onto the grid
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
    return y_grid


def cutoff_ind(data):
    for i in range(0, len(data)):

        if data.loc[i, "margin_1"] < 0:
            data.loc[i, "cutoff_ind"] = 0

        elif data.loc[i, "margin_1"] > 0:
            data.loc[i, "cutoff_ind"] = 1

    return data


def ttest_uncool(data):
    table = pd.DataFrame(
        {'Municipaliy Characteristic': [], 'Female Mayor': [], 'Male Mayor': [], 'Diff': [], 'Std. Error': [],
         'Observations': [],
         'P-Value (Extension)': []})
    table = table.set_index(['Municipaliy Characteristic'])

    col_0 = data[0].columns
    col_1 = data[1].columns

    for i in range(0, len(col_0)):
        x_1 = data[0][col_0[i]]
        y_1 = data[1][col_1[i]]
        m_1 = data[0][col_0[i]].mean().round(3)
        m_2 = data[1][col_1[i]].mean().round(3)
        diff = (m_1 - m_2).round(3)
        n_1 = len(data[0])
        n_2 = len(data[1])
        n = n_1 + n_2
        std_1 = np.std(data[0][col_0[i]])
        std_1 = np.asarray(std_1)
        std_2 = np.std(data[1][col_1[i]])
        std_2 = np.asarray(std_2)
        std_error = (se(n_1, n_2, std_1, std_2)).round(3)
        # se(n_1,n_2,std_x1,std_y1)
        test = sm.stats.weightstats.ttest_ind(x_1, y_1, usevar='pooled')

        if test[1] <= 0.01:
            diff_res = str(diff) + "***"
        elif 0.01 < test[1] <= 0.05:
            diff_res = str(diff) + "**"
        elif 0.05 < test[1] <= 0.1:
            diff_res = str(diff) + "*"
        else:
            diff_res = str(diff)

        output = [m_1, m_2, diff_res, std_error, n, test[1].round(3)]

        table.loc['' + str(col_0[i]) + ''] = output
    return table


def ttest(data, index):
    table = pd.DataFrame(
        {index: [], 'Treatment': [], 'Control ': [], 'Diff': [], 'Std. Error': [],
         'Observations': []})
    table = table.set_index([index])

    col_0 = data[0].columns
    col_1 = data[1].columns

    for i in range(0, len(col_0)):
        x_1 = data[0][col_0[i]].dropna()
        y_1 = data[1][col_1[i]].dropna()
        m_1 = data[0][col_0[i]].mean().round(3)
        m_2 = data[1][col_1[i]].mean().round(3)
        diff = (m_1 - m_2).round(3)
        n_1 = len(data[0][col_0[i]].dropna())
        n_2 = len(data[1][col_1[i]].dropna())
        n = n_1 + n_2
        std_1 = np.std(data[0][col_0[i]])
        std_1 = np.asarray(std_1)
        std_2 = np.std(data[1][col_1[i]])
        std_2 = np.asarray(std_2)
        std_error = (se(n_1, n_2, std_1, std_2)).round(3)
        # se(n_1,n_2,std_x1,std_y1)
        test = sm.stats.weightstats.ttest_ind(x_1, y_1, usevar='pooled')

        if test[1] <= 0.01:
            diff_res = str(diff) + "***"
        elif 0.01 < test[1] <= 0.05:
            diff_res = str(diff) + "**"
        elif 0.05 < test[1] <= 0.1:
            diff_res = str(diff) + "*"
        else:
            diff_res = str(diff)

        output = [m_1, m_2, diff_res, std_error, n]

        table.loc['' + str(col_0[i]) + ''] = output
    return table


def t_test_prepare_cha(data):
    table_f = data[data["geschl_first_placed"] == "f"].drop(columns=data.columns[range(0, 12)])
    table_m = data[data["geschl_first_placed"] == "m"].drop(columns=data.columns[range(0, 12)])

    table_f = table_f.rename(columns={'log_bevoelkerung': 'Log(population)', 'log_flaeche': 'Log(land area)',
                                      'log_debt_pc': 'Log(debt p.c.)', 'log_tottaxrev_pc': 'Log(tax revenues p.c.)',
                                      'log_gemeinde_beschaef_pc': 'Log(local gov. employment p.c.)',
                                      'log_female_sh_gem_besch': 'Log(female share, local gov. employment)',
                                      'log_tot_beschaeft_pc': 'Log(total employment p.c.)',
                                      'log_female_share_totbesch': 'Log(female share, total employment)',
                                      'log_prod_share_tot': 'Log(manufacturing / total employment)',
                                      'log_female_share_prod': 'Log(female share, manufacturing'})
    table_m = table_m.rename(columns={'log_bevoelkerung': 'Log(population)', 'log_flaeche': 'Log(land area)',
                                      'log_debt_pc': 'Log(debt p.c.)', 'log_tottaxrev_pc': 'Log(tax revenues p.c.)',
                                      'log_gemeinde_beschaef_pc': 'Log(local gov. employment p.c.)',
                                      'log_female_sh_gem_besch': 'Log(female share, local gov. employment)',
                                      'log_tot_beschaeft_pc': 'Log(total employment p.c.)',
                                      'log_female_share_totbesch': 'Log(female share, total employment)',
                                      'log_prod_share_tot': 'Log(manufacturing / total employment)',
                                      'log_female_share_prod': 'Log(female share, manufacturing)'})

    return table_f, table_m


def t_test_prepare_rank(data):
    table_f = data[data["female"] == 1].drop(
        columns=['gkz', 'jahr', 'gkz_jahr', 'rdd_sample', 'female', 'elected', 'gewinn', 'gewinn_dummy', 'joint_party',
                 'age', 'non_university_phd', 'university', 'phd',
                 'architect', 'businessmanwoman', 'engineer', 'lawyer', 'civil_administration', 'teacher', 'employed',
                 'selfemployed', 'student', 'retired', 'housewifehusband', 'incumbent_council', 'wahlbet', 'party',
                 'female_mayor', 'margin_1', 'inter_1', 'margin_2',
                 'inter_2', 'female_mayor_full_sample', 'sum_years_as_mayor', 'mayor_age', 'mayor_university',
                 'mayor_employment', 'log_bevoelkerung', 'log_flaeche', 'log_debt_pc', 'log_tottaxrev_pc',
                 'log_gemeinde_beschaef_pc', 'log_female_sh_gem_besch', 'log_tot_beschaeft_pc',
                 'log_female_share_totbesch', 'log_prod_share_tot', 'log_female_share_prod'])

    table_m = data[data["female"] == 0].drop(
        columns=['gkz', 'jahr', 'gkz_jahr', 'rdd_sample', 'female', 'elected', 'gewinn', 'gewinn_dummy', 'joint_party',
                 'age', 'non_university_phd', 'university', 'phd',
                 'architect', 'businessmanwoman', 'engineer', 'lawyer', 'civil_administration', 'teacher', 'employed',
                 'selfemployed', 'student', 'retired', 'housewifehusband', 'incumbent_council', 'wahlbet', 'party',
                 'female_mayor', 'margin_1', 'inter_1', 'margin_2',
                 'inter_2', 'female_mayor_full_sample', 'sum_years_as_mayor', 'mayor_age', 'mayor_university',
                 'mayor_employment', 'log_bevoelkerung', 'log_flaeche', 'log_debt_pc', 'log_tottaxrev_pc',
                 'log_gemeinde_beschaef_pc', 'log_female_sh_gem_besch', 'log_tot_beschaeft_pc',
                 'log_female_share_totbesch', 'log_prod_share_tot', 'log_female_share_prod'])

    table_f = table_f.rename(
        columns={'gewinn_norm': 'Rank improvement (normalized)', 'listenplatz_norm': 'Initial list rank (normalized)'})
    table_m = table_m.rename(
        columns={'gewinn_norm': 'Rank improvement (normalized)', 'listenplatz_norm': 'Initial list rank (normalized)'})
    return table_f, table_m


def t_test_prepare_party(data):
    table_f = data[data["geschl_first_placed"] == "f"].drop(
        columns=['gkz', 'jahr', 'election_year', 'mayor_election_year', 'geschl_first_placed', 'rdd_sample',
                 'female_mayor', 'male_mayor', 'margin_1'])
    table_m = data[data["geschl_first_placed"] == "m"].drop(
        columns=['gkz', 'jahr', 'election_year', 'mayor_election_year', 'geschl_first_placed', 'rdd_sample',
                 'female_mayor', 'male_mayor', 'margin_1'])
    table_f = table_f.rename(columns={'cdu_winner': 'CDU', 'spd_winner': 'SPD',
                                      'other_party_winner': 'Other'})
    table_m = table_m.rename(columns={'cdu_winner': 'CDU', 'spd_winner': 'SPD',
                                      'other_party_winner': 'Other'})
    return table_f, table_m


def t_test_prepare_can(data):
        data_temp = data.drop(
            columns=['age', 'gkz', 'jahr', 'gkz_jahr', 'rdd_sample', 'female', 'elected', 'gewinn_norm', 'gewinn',
                     'gewinn_dummy', 'listenplatz_norm', 'joint_party', 'incumbent_council', 'wahlbet', 'party',
                     'margin_1', 'inter_1', 'margin_2', 'inter_2',
                     'female_mayor_full_sample', 'sum_years_as_mayor', 'mayor_age', 'mayor_university',
                     'mayor_employment', 'log_bevoelkerung', 'log_flaeche', 'log_debt_pc', 'log_tottaxrev_pc',
                     'log_gemeinde_beschaef_pc', 'log_female_sh_gem_besch', 'log_tot_beschaeft_pc',
                     'log_female_share_totbesch', 'log_prod_share_tot', 'log_female_share_prod'])

        table_c = data_temp[data_temp["female_mayor"] == 0].drop(columns=['female_mayor'])

        table_c = table_c.rename(
            columns={'non_university_phd': 'Highschool', 'university': 'University Diploma', 'phd': 'PhD Degree',
                     'architect': 'Architect', 'businessmanwoman': "Businesswoman/-man", 'engineer': "Engineer",
                     'lawyer': "Lawyer", 'civil_administration': "Civil administration", "teacher": "Teacher",
                     'employed': "Employed", 'selfemployed': "Self-employed", "student": "Student",
                     'retired': "Retired",
                     'housewifehusband': "Housewife/-husband"})

        table_t = data_temp[data_temp["female_mayor"] == 1].drop(columns=['female_mayor'])

        table_t = table_t.rename(
            columns={'non_university_phd': 'Highschool', 'university': 'University Diploma', 'phd': 'PhD Degree',
                     'architect': 'Architect', 'businessmanwoman': "Businesswoman/-man", 'engineer': "Engineer",
                     'lawyer': "Lawyer", 'civil_administration': "Civil administration", "teacher": "Teacher",
                     'employed': "Employed", 'selfemployed': "Self-employed", "student": "Student",
                     'retired': "Retired",
                     'housewifehusband': "Housewife/-husband"})

        return table_t, table_c


def tt_test_subsamples(*inputs):
    table_f = pd.DataFrame()
    table_m = pd.DataFrame()
    for i in inputs:
        ttest_f = i[0]
        ttest_m = i[0]
        ttest_f = np.asarray(i[0].loc[(i[0]["geschl_first_placed"] == "f")][i[1]])
        ttest_m = np.asarray(i[0].loc[(i[0]["geschl_first_placed"] == "m")][i[1]])
        table_f["DF_" + str(i[1]) + "_f"] = ttest_f
        table_m["DF_" + str(i[1]) + "_m"] = ttest_m
    return table_f, table_m


def se(n_1, n_2, std_1, std_2):
    error = np.sqrt((((n_1 - 1) * (std_1 ** 2) + (n_2 - 1) * (std_2 ** 2)) / (n_1 + n_2 - 2)) * (1 / n_1 + 1 / n_2))
    return error
