""" This module contains  utility  and table output functions for the replication notebook"""

import pandas as pd
import numpy as np
import statsmodels.api as sm_api
import statsmodels as sm
import warnings

from localreg import *

warnings.filterwarnings("ignore")


def ind_fct(vector):
    """ Indicator function that returns a list of len(vector) indicating if value in vector is positive/negative.

    :param (list)       vector: list containing values to check
    :return:            list of integers (0,1)
    """

    indic = [1 if abs(x) <= 1 else 0 for x in vector]

    return indic


def weight(data, bw):
    """Calculates weights for observations based on distance to cut-off point ("margin_1"=0) for regression model.

    :param df           data: dataframe containing variable "margin_1"
    :param (int)        bw: distance to cut-off
    :return:            df containing weights as variable
    """

    temp_i = data['margin_1'] / bw
    ind_i = ind_fct(temp_i)
    weight_i = (1 - abs(temp_i)) * ind_i
    data.loc[:, ('weight' + str(bw) + '')] = weight_i


def table1(data, dic, female=False):
    """  This function produces summary statistics for individual council candidates  as presented in Table 1/section 6.

    :param (df)         data: dataframe containing variable "margin_1"
    :param (dic)        dic: dictionary mapping variable names to output names
    :param (bool)       female: if True, restricts the sample to female council candidates
    :return:            df containing summary statistics
    """

    if female:
        data = data[data["female"] == 1]

    data = data[dic.keys()].rename(columns=dic)
    table1_a = data.describe(percentiles=[]).transpose()[["count", "mean", "std", "min", "max"]]
    return table1_a


def subset_by_margin(data, margin):
    """This function restricts the dataframe to only contain observations with margin / distance to cut-off point
    "margin_1"=0.

    :param (df)         data: dataframe containing variable "margin_1"
    :param (float)      margin: maximum distance to cut-off point
    :return:            restricted df
    """
    data = data[abs(data["margin_1"]) < margin]

    return data


def se(n_1, n_2, std_1, std_2):
    """This function calculates an estimator of the pooled standard error for two samples of unequal size.

    :param (int)        n_1: size of sample 1
    :param (int)        n_2: size of sample 2
    :param (np.ndarray) std_1: standard deviation of sample 1
    :param (np.ndarray) std_2: standard deviation of sample 2
    :return:(np.float)  estimator of pooled standard error
    """
    error = np.sqrt((((n_1 - 1) * (std_1 ** 2) + (n_2 - 1) * (std_2 ** 2)) / (n_1 + n_2 - 2)) * (1 / n_1 + 1 / n_2))
    return error


def significance_level(value_to_check, string_to_print):
    """ This function attaches ***/**/* to float if float meets conditions.
    Designed to be used in ttest(), reg_tab and reg_tab_ext to indicate statistical significance.

    :param (float)      value_to_check: value to check
    :param (str)        string_to_print: string ***/**/* gets attached to if condition is met
    :return:            string
    """
    if value_to_check <= 0.01:
        out = str(string_to_print) + "***"
    elif 0.01 < value_to_check <= 0.05:
        out = str(string_to_print) + "**"
    elif 0.05 < value_to_check <= 0.1:
        out = str(string_to_print) + "*"
    else:
        out = str(string_to_print)

    return out


def ttest(data, index):
    """Performs t-test for differences between variables for two samples.

    :param tuple        data:  tuple containing two dataframes with identical variables which are to be compared
    :param str          index: overall label for characteristics in table
    :return:            df containing results of t-tests
    """
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


def reg_tab(*model):
    """ Performs weighted linear regression for various models as specified in section 4 of the replication notebook.
        A single model (i.e. function argument) takes on the form:

            model=[df,polynomial, bw, dependant variable, bandwidth-type]

        df: dataframe containing all relevant data
        polynomial (str): "quadratic" includes quadratic values of "margin_1" and "inter_1" in regression model;
            default is "linear"
        bw (float): specifying data to be included relative to cut-off point ("margin_1"=0)
        dependant variable (str): name of dependant variable
        bandwidth-type (str): method used to calculate bandwidth

    :return: df containing results of regression model
    """
    table = pd.DataFrame(
        {'Model': [], 'Female Mayor': [], 'Std.err': [], 'Bandwidth type': [], 'Bandwidth size': [], 'Polynomial': [],
         'Observations': [], 'Elections': [], 'Municipalities': [],
         'Mean': [], 'Std.err (Mean)': []})

    table = table.set_index(['Model'])

    for counter, i in enumerate(model):

        data_i = subset_by_margin(i[0], i[2])
        weight(data_i, i[2])

        y = data_i[i[3]]
        w = data_i["weight" + str(i[2]) + ""]

        x = data_i[["female_mayor", "margin_1", "inter_1"]]

        polynomial_i = str("Linear")

        if i[1] == "quadratic":
            x = data_i[["female_mayor", "margin_1", "inter_1", "margin_2", "inter_2"]]

            polynomial_i = str("Quadratic")

        x = sm_api.add_constant(x)
        wls_model = sm_api.WLS(y, x, weights=w)
        results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': data_i["gkz"]})

        result_parameter = significance_level(results.pvalues["female_mayor"], results.params[1].round(3))

        bw_size_i = str(round(i[2], 2))
        bw_type_i = str(i[4])

        output = [result_parameter, results.bse[1], bw_type_i, bw_size_i, polynomial_i, results.nobs,
                  data_i["gkz_jahr"].value_counts().count(),
                  data_i["gkz"].value_counts().count(), y.mean().round(2), np.std(y)]

        table.loc["(" + str(counter + 1) + ")"] = output
    table = table.round(3)
    return table


def reg_tab_ext(*model):
    """ Performs weighted linear regression for various models building upon the model specified in section 4,
        while additionally including education levels of a council candidate (university degree, doctoral/PhD degree)
        A single model (i.e. function argument) takes on the form:


            model=[df,polynomial, bw, dependant variable, bandwidth-type]

    df: dataframe containing all relevant data
    polynomial (str): "quadratic" includes quadratic values of "margin_1" and "inter_1" in regressionmodel;
        default is "linear"
    bw (float): specifying data to be included relative  to  cut-off point ("margin_1"=0)
    dependant variable (str): name of dependant variable
    bandwidth-type (str): method used to calculate bandwidth

    :return: df containing results of regression
    """
    # pd.set_option('mode.chained_assignment', None)
    table = pd.DataFrame(
        {'Model': [], 'Female Mayor': [], 'Std.err_Female Mayor': [], 'University': [], 'Std.err_University': [],
         'PhD': [], 'Std.err_PhD': [], 'Bandwidth type': [], 'Bandwidth size': [], 'Polynomial': [],
         'Observations': [], 'Elections': [], 'Municipalities': [],
         'Mean': [], 'Std.err (Mean)': []})

    table = table.set_index(['Model'])

    for counter, i in enumerate(model):

        data_i = subset_by_margin(i[0], i[2])
        weight(data_i, i[2])

        y = data_i[i[3]]
        w = data_i["weight" + str(i[2]) + ""]

        x = data_i[["female_mayor", "margin_1", "inter_1", 'university', 'phd']]
        polynomial_i = str("Linear")

        if i[1] == "quadratic":
            x = data_i[["female_mayor", "margin_1", "inter_1", 'university', 'phd', "margin_2", "inter_2"]]
            polynomial_i = str("Quadratic")

        x = sm_api.add_constant(x)
        wls_model = sm_api.WLS(y, x, missing='drop', weights=w)
        results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': data_i["gkz"]})

        betas = [1, 2, 3]
        cov = ["female_mayor", 'university', 'phd']

        for j in cov:
            betas[cov.index(j)] = significance_level(results.pvalues[j], results.params[(cov.index(j) + 1)].round(3))

        bw_size_i = str(round(i[2], 2))
        bw_type_i = str(i[4])

        output = [betas[0], results.bse[1], betas[1], results.bse[4], betas[2], results.bse[5], bw_type_i, bw_size_i,
                  polynomial_i, results.nobs,
                  data_i["gkz_jahr"].value_counts().count(),
                  data_i["gkz"].value_counts().count(), y.mean().round(2), np.std(y)]

        table.loc["(" + str(counter + 1) + ")"] = output
    table = table.round(3)

    return table
