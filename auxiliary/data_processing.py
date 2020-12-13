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


def dictionary_table1():
    dic = {"gewinn_norm": "Rank improvement (normalized)", "listenplatz_norm": "Initial list rank (normalized)",
           "age": "Age", 'non_university_phd': "High school", 'university': 'University', 'phd': 'Phd',
           'architect': 'Architect', 'businessmanwoman': "Businesswoman/-man", 'engineer': "Engineer",
           'lawyer': "Lawyer", 'civil_administration': "Civil administration", "teacher": "Teacher",
           'employed': "Employed", 'selfemployed': "Self-employed", "student": "Student", 'retired': "Retired",
           'housewifehusband': "Housewife/-husband"}
    return dic

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


def get_rdd_data(data, female=False):
    # creates sample for RDD

    if female:
        data1 = data.loc[(data["rdd_sample"] == 1) & (data["female"] == 1)].copy()

        return data1

    data1 = data.loc[(data["rdd_sample"] == 1)].copy()
    return data1
