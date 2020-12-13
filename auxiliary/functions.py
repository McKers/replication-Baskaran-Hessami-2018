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
from auxiliary.data_processing import *
from auxiliary.functions import *
from auxiliary.plots import *


def ind_fct(vector):
    indic = [1 if abs(x) <= 1 else 0 for x in vector]

    return indic


def weight(data, bw):
    temp_i = data['margin_1'] / bw
    ind_i = ind_fct(temp_i)
    weight_i = (1 - abs(temp_i)) * ind_i
    data.loc[:, ('weight' + str(bw) + '')] = weight_i


def table1(data, dic, female=False):
    if female:
        data = data[data["female"] == 1]

    data = data[dic.keys()].rename(columns=dic)
    table1_a = data.describe(percentiles=[]).transpose()[["count", "mean", "std", "min", "max"]]
    return table1_a


def reg_tab(*model):
    table = pd.DataFrame(
        {'Model': [], 'Female Mayor': [], 'Std.err': [], 'Bandwidth type': [], 'Bandwidth size': [], 'Polynominal': [],
         'Observations': [], 'Elections': [], 'Municipalities': [],
         'Mean': [], 'Std.err (Mean)': []})

    table = table.set_index(['Model'])

    for counter, i in enumerate(model):

        data_i = subset_by_margin(i[0], i[2])
        weight(data_i, i[2])

        y = data_i[i[3]]
        w = data_i["weight" + str(i[2]) + ""]

        x = data_i[["female_mayor", "margin_1", "inter_1"]]

        polynominal_i = str("Linear")

        if i[1] == "quadratic":
            x = data_i[["female_mayor", "margin_1", "inter_1", "margin_2", "inter_2"]]

            polynominal_i = str("Quadratic")

        x = sm_api.add_constant(x)
        wls_model = sm_api.WLS(y, x, weights=w)
        results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': data_i["gkz"]})

        result_parameter = significance_level(results.pvalues["female_mayor"], results.params[1].round(3))

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
        data_i = subset_by_margin(i[0], i[2])
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


def display_side_by_side(*args):
    # credit goes to stackoverflow user: https://stackoverflow.com/users/508907/ntg
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


def subset_by_margin(data, margin):
    data = data[abs(data["margin_1"]) < margin]

    return data


def se(n_1, n_2, std_1, std_2):
    error = np.sqrt((((n_1 - 1) * (std_1 ** 2) + (n_2 - 1) * (std_2 ** 2)) / (n_1 + n_2 - 2)) * (1 / n_1 + 1 / n_2))
    return error


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
        std_1 = np.asarray(np.std(data[0][col_0[i]]))
        std_2 = np.asarray(np.std(data[1][col_1[i]]))

        std_error = (se(n_1, n_2, std_1, std_2)).round(3)
        test = sm.stats.weightstats.ttest_ind(x_1, y_1, usevar='pooled')

        diff_res = significance_level(test[1], diff)

        output = [m_1, m_2, diff_res, std_error, n]

        table.loc['' + str(col_0[i]) + ''] = output

    return table


def significance_level(value_to_check, string_to_print):
    if value_to_check <= 0.01:
        out = str(string_to_print) + "***"
    elif 0.01 < value_to_check <= 0.05:
        out = str(string_to_print) + "**"
    elif 0.05 < value_to_check <= 0.1:
        out = str(string_to_print) + "*"
    else:
        out = str(string_to_print)

    return out
