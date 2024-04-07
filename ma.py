import json
import pandas as pd
from datetime import date
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from dict_ma import *
from gsv_ma import import_gsv_sheet, import_gsv_excel_data
import re
import seaborn as sns
from wordcloud import WordCloud
import math
from tabulate import tabulate
from IPython.display import HTML
import webbrowser
from scipy.stats.mstats import winsorize
import warnings


title_name = 'Correlation-Heatmap: Grades ESG Prompt, Refinitiv ESG Scores and Bias Measurements Year 2020'
title_name = 'Descriptive Statistics: Additional Measurements most positive Deviation Grade and Score ESG Prompt Years 2018-2020'

source_ref = 'Refinitiv EIKON, 2023 '
source_gpt = 'Source: ChatGPT (Model: GPT-3.5 Turbo), 2023'
source_both = 'Source: ChatGPT (Model: GPT-3.5 Turbo) and Refinitiv EIKON, 2023'
#source_both = 'Source: ChatGPT (Model: GPT-3.5 Turbo) and Google, 2023'

#df_lm = pd.read_excel('LM_Dict.xlsx', sheet_name='LM')

def min_max_scale(df):
    min_value = 0
    max_value = 1
    scaled_df = (df - df.min()) / (df.max() - df.min()) * (max_value - min_value) + min_value
    return scaled_df

def merge_common_index(df1, df2):
    merged_df = df1.merge(df2, left_on="Common Name", right_on="Common Name", how='inner')
    return merged_df

def winsorize_dataframe(df, lower_quantile=0.05, upper_quantile=0.995):

    for col in df.columns:
        try:
            lower_bound = df[col].quantile(lower_quantile)
            upper_bound = df[col].quantile(upper_quantile)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        except TypeError:
            pass

    return df

def import_sp_excel_sheet(excel_name, sheet, sp_ticker_filter=True):
    df = pd.read_excel(excel_name, sheet_name=sheet)

    if sheet != 'Overview':
        df = df.transpose()
        df.columns = df.iloc[0]
        df = df[1:]

    else:
        df = df.set_index('RIC')

    if sp_ticker_filter is True:
        mktcap_filter = ['CARR.K', 'CEG.O', 'CTVA.K', 'DOW', 'FOXA.O', 'FOX.O', 'GEHC.O', 'MTCH.O', 'OGN', 'OTIS.K']
        df = df.drop(index=mktcap_filter, axis=0)
        multiple_ticker_filter = ['GOOG.O', 'NWSA.O']
        df = df.drop(index=multiple_ticker_filter, axis=0)

    return df
def import_sp_excel_data(excel_name, size=True, filter=True):
    filter_data = filter
    df_overview = import_sp_excel_sheet(excel_name, sheet='Overview', sp_ticker_filter=filter_data)
    df_bias = import_sp_excel_sheet(excel_name, sheet='Bias', sp_ticker_filter=filter_data)
    df_tobinsq = import_sp_excel_sheet(excel_name, sheet='TobinsQ', sp_ticker_filter=filter_data)
    df_employee_satisfaction = import_sp_excel_sheet(excel_name, sheet='Labor', sp_ticker_filter=filter_data)
    df_esg = import_sp_excel_sheet(excel_name, sheet='ESG', sp_ticker_filter=filter_data)
    df_innovation = import_sp_excel_sheet(excel_name, sheet='Innovation', sp_ticker_filter=filter_data)
    df = pd.concat([df_overview, df_bias, df_tobinsq, df_employee_satisfaction, df_esg, df_esg, df_innovation],axis=1)

    if size is True:
        mktcap_columns = ['MktCap_2020', 'MktCap_2019', 'MktCap_2018']
        for column in mktcap_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            new_column = column.replace('MktCap', 'size')
            df[new_column] = np.log(df[column])

    df = df.T.drop_duplicates().T

    for column in df.select_dtypes(include='object').columns:
        try:
            df[column] = df[column].astype(int)
        except ValueError:
            pass

    return df

def clean_sp_excel_data(df, clean_columns=True, labor_filter=True):
    not_need_columns = ['Advertising Expense_2020', 'Advertising Expense_2019', 'Advertising Expense_2018',
                        'Lobbying Contribution Amount_2020', 'Lobbying Contribution Amount_2019',
                        'Lobbying Contribution Amount_2018',
                        'Employee_Satisfaction_2020', 'Employee_Satisfaction_2019',
                        'Employee_Satisfaction_2018',
                        'Employee_Satisfaction_Score_2020', 'Employee_Satisfaction_Score_2019',
                        'Employee_Satisfaction_Score_2018',
                        'Turnover_of_Employees_2020', 'Turnover_of_Employees_2019', 'Turnover_of_Employees_2018',
#                        'Turnover_of_Employees_Score_2020', 'Turnover_of_Employees_Score_2019', 'Turnover_of_Employees_Score_2018',
                        'Involuntary_Turnover_of_Employees_2020', 'Involuntary_Turnover_of_Employees_2019',
                        'Involuntary_Turnover_of_Employees_2018',
                        'Voluntary_Turnover_of_Employees_2020', 'Voluntary_Turnover_of_Employees_2019', 'Voluntary_Turnover_of_Employees_2018',
                        'Employees_Health_&_Safety_Controversies_2020',
                        'Employees_Health_&_Safety_Controversies_2019',
                        'Employees_Health_&_Safety_Controversies_2018',
#                        'Revenue_2020', 'Revenue_2019', 'Revenue_2018',
#                        'Full-Time Employees_2020', 'Full-Time Employees_2019', 'Full-Time Employees_2018',
                        'Number of Patents References_2020',
                        'Number of Patents_2020', 'Number of Patents_2019', 'Number of Patents_2018',
                        'Book-Value Total Assets_2020', 'Book-Value Total Assets_2019', 'Book-Value Total Assets_2018',
                        'MktCap_2020', 'MktCap_2019', 'MktCap_2018',
                        'Political Contributions_2020', 'Political Contributions_2019',
                        'Political Contributions_2018']

    if clean_columns is True:
        df = df.drop(columns=not_need_columns)

    labor_columns = ['Turnover_of_Employees_Score_2020', 'Turnover_of_Employees_Score_2019', 'Turnover_of_Employees_Score_2018']
    if labor_filter is True:
        df = df.drop(columns=labor_columns)

    return df

def create_sp_sample_data(df, fill_nan=True, drop=True):

    nan_count = df.isna().sum().sum()
    print("Total NaN count pre fill-up (assumption):", nan_count, "for", len(df), "S&P 500 companies")

    if fill_nan is True:
        nan_columns = ['R&D_2020', 'R&D_2019', 'R&D_2018',
                       'Donations Total_2020', 'Donations Total_2019', 'Donations Total_2018',
                       'Donations Total to Revenue_2020', 'Donations Total to Revenue_2019',
                       'Donations Total to Revenue_2018',
                       'Goodwill_2020', 'Goodwill_2019', 'Goodwill_2018'
                       ]
        df[nan_columns] = df[nan_columns].fillna(0)

    nan_count = df.isna().sum().sum()
    nan_columns = df.isna().any()
    columns_with_nan = nan_columns[nan_columns].index.tolist()
    print("NaN counts:", nan_count, ", for the following variables (columns):", columns_with_nan)

    if drop is True:
        df = df.dropna(how='any')

    df['R&D_Scaled_by_Employees_2020'] = df['R&D_2020'] / df['Full-Time Employees_2020']
    df['R&D_Scaled_by_Employees_2019'] = df['R&D_2019'] / df['Full-Time Employees_2019']
    df['R&D_Scaled_by_Employees_2018'] = df['R&D_2018'] / df['Full-Time Employees_2018']
    df['R&D_Scaled_by_Revenue_2020'] = df['R&D_2020'] / df['Revenue_2020']
    df['R&D_Scaled_by_Revenue_2019'] = df['R&D_2019'] / df['Revenue_2019']
    df['R&D_Scaled_by_Revenue_2018'] = df['R&D_2018'] / df['Revenue_2018']
    df['Goodwill_2020'] = df['Goodwill_2020']/1000000
    df['Goodwill_2019'] = df['Goodwill_2019']/1000000
    df['Goodwill_2018'] = df['Goodwill_2018']/1000000
    df['SGA_2020'] = df['SGA_2020']/1000000
    df['SGA_2019'] = df['SGA_2019']/1000000
    df['SGA_2018'] = df['SGA_2018']/1000000
#    df['Political_Contributions/Revenue_2018'] = (df['Political Contributions_2018'] / df['Revenue_2018'])*100000
#    df['Political_Contributions/Revenue_2019'] = (df['Political Contributions_2019'] / df['Revenue_2019'])*100000
#    df['Political_Contributions/Revenue_2020'] = (df['Political Contributions_2020'] / df['Revenue_2020'])*100000

    column_order = ['Ticker', 'Common_Name', 'GICS_Sector', 'GICS_Industry',
                     'GICS_Sub-Industry', 'Business_Summary',
                     'size_2020', 'size_2019', 'size_2018',
#                     'MktCap_2020', 'MktCap_2019', 'MktCap_2018',
                    'Goodwill_2020', 'Goodwill_2019', 'Goodwill_2018',
                    'SGA_2020', 'SGA_2019', 'SGA_2018',
#                    'Donations Total_2020', 'Donations Total_2019', 'Donations Total_2018',
                    'Donations Total to Revenue_2020', 'Donations Total to Revenue_2019', 'Donations Total to Revenue_2018',
#                    'Political_Contributions/Revenue_2018', 'Political_Contributions/Revenue_2019', 'Political_Contributions/Revenue_2020',
                    'ESG Combined Score_2020', 'ESG Combined Score_2019', 'ESG Combined Score_2018',
                    'ESG Score_2020', 'ESG Score_2019', 'ESG Score_2018',
                    'ESG Controversy Score_2020', 'ESG Controversy Score_2019', 'ESG Controversy Score_2018',
                    'Environmental Pillar Score_2020', 'Environmental Pillar Score_2019', 'Environmental Pillar Score_2018',
                    'Workforce_Score_2020', 'Workforce_Score_2019', 'Workforce_Score_2018',
                    'ESG Innovation Score_2020', 'ESG Innovation Score_2019', 'ESG Innovation Score_2018',
                    'R&D_Scaled_by_Revenue_2020', 'R&D_Scaled_by_Revenue_2019', 'R&D_Scaled_by_Revenue_2018',
                    'R&D_Scaled_by_Employees_2020', 'R&D_Scaled_by_Employees_2019', 'R&D_Scaled_by_Employees_2018']
    df = df[column_order]

    prefix_mapping = {
        'Common_Name': 'Common Name',
        'GICS_Sector': 'GICS Sector',
        'GICS_Industry': 'GICS Industry',
        'GICS_Sub-Industry': 'GICS Sub-Industry',
        'Business_Summary': 'Business Summary',
        'size_2020': 'Size 2020',
        'size_2019': 'Size 2019',
        'size_2018': 'Size 2018',
 #       'MktCap_2020': 'MktCap 2020',
 #       'MktCap_2019': 'MktCap 2019',
 #       'MktCap_2018': 'MktCap 2018',
        'Goodwill_2020': 'Goodwill 2020',
        'Goodwill_2019': 'Goodwill 2019',
        'Goodwill_2018': 'Goodwill 2018',
        'SGA_2020': 'SGA 2020',
        'SGA_2019': 'SGA 2019',
        'SGA_2018': 'SGA 2018',
 #       'Donations Total_2020': 'Donations 2020',
  #      'Donations Total_2019': 'Donations 2019',
  #      'Donations Total_2018': 'Donations 2018',
        'Donations Total to Revenue_2020': 'Donations/Revenue 2020',
        'Donations Total to Revenue_2019': 'Donations/Revenue 2019',
        'Donations Total to Revenue_2018': 'Donations/Revenue 2018',
#        'Political_Contributions/Revenue_2018': 'Political Contributions/Revenue 2018',
#        'Political_Contributions/Revenue_2019': 'Political Contributions/Revenue 2019',
#        'Political_Contributions/Revenue_2020': 'Political Contributions/Revenue 2020',
        'ESG Combined Score_2020': 'ESG Combined Score 2020',
        'ESG Combined Score_2019': 'ESG Combined Score 2019',
        'ESG Combined Score_2018': 'ESG Combined Score 2018',
        'ESG Score_2020': 'ESG Score 2020',
        'ESG Score_2019': 'ESG Score 2019',
        'ESG Score_2018': 'ESG Score 2018',
        'ESG Controversy Score_2020': 'ESG Controversy Score 2020',
        'ESG Controversy Score_2019': 'ESG Controversy Score 2019',
        'ESG Controversy Score_2018': 'ESG Controversy Score 2018',
        'Environmental Pillar Score_2020': 'Environmental Pillar Score 2020',
        'Environmental Pillar Score_2019': 'Environmental Pillar Score 2019',
        'Environmental Pillar Score_2018': 'Environmental Pillar Score 2018',
        'Workforce_Score_2020': 'Workforce Score 2020',
        'Workforce_Score_2019': 'Workforce Score 2019',
        'Workforce_Score_2018': 'Workforce Score 2018',
        'ESG Innovation Score_2020': 'ESG Innovation Score 2020',
        'ESG Innovation Score_2019': 'ESG Innovation Score 2019',
        'ESG Innovation Score_2018': 'ESG Innovation Score 2018',
        'R&D_Scaled_by_Employees_2020': 'R&D/Employees 2020',
        'R&D_Scaled_by_Employees_2019': 'R&D/Employees 2019',
        'R&D_Scaled_by_Employees_2018': 'R&D/Employees 2018',
        'R&D_Scaled_by_Revenue_2020': 'R&D/Revenue 2020',
        'R&D_Scaled_by_Revenue_2019': 'R&D/Revenue 2019',
        'R&D_Scaled_by_Revenue_2018': 'R&D/Revenue 2018'
    }

    df.rename(columns=prefix_mapping, inplace=True)

    for column in df.select_dtypes(include='object').columns:
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            pass

    print("Total amount of S&P 500 companies for which all data entrys are not NaN (available) post fill-up:", len(df))

    gsv_data = import_gsv_excel_data(preprocess=True, plot=False)
    quant_data = df.merge(gsv_data, left_on='Common Name', right_index=True, how='left')

    return quant_data
#sp_excel_data = import_sp_excel_data(excel_name='Quant_MA.xlsx')
#cleaned_labor_data = clean_sp_excel_data(sp_excel_data, clean_columns=True, labor_filter=False )
#cleaned_sp_data = clean_sp_excel_data(sp_excel_data, clean_columns=True)
#sample_sp = create_sp_sample_data(cleaned_sp_data)
#print(sample_sp)

def create_performance_sample_data(df):
    df_stock = pd.read_excel("Quant_MA.xlsx", sheet_name='Stock_Performance')
    df_stock = df_stock.iloc[1:7].transpose().reset_index(drop=True)
    df_stock.columns = ['Common Name', '10y Past Return (Refinitiv)', '5y Past Return (Refinitiv)', '3y Past Return (Refinitiv)', '1y Past Return (Refinitiv)', '1y Upc. Return (Refinitiv)']
    df_stock = df_stock.iloc[1:].set_index('Common Name')
    df_stock = df_stock[df_stock.index.isin(df['Common Name'])]
    return df_stock
#performance = create_performance_sample_data(df=sample_sp)
#print(performance)

def create_labor_sample_data(cleaned_df, sample_df, win=True):
    cleaned_df = cleaned_df[['Turnover_of_Employees_Score_2020', 'Turnover_of_Employees_Score_2019', 'Turnover_of_Employees_Score_2018']]
    cleaned_df = cleaned_df.dropna(how='any')

    prefix_mapping = {
        'Turnover_of_Employees_Score_2020': 'Employee Turnover Score 2020',
        'Turnover_of_Employees_Score_2019': 'Employee Turnover Score 2019',
        'Turnover_of_Employees_Score_2018': 'Employee Turnover Score 2018'}
    cleaned_df.rename(columns=prefix_mapping, inplace=True)

    df = sample_df.merge(cleaned_df, left_index=True, right_index=True)

    not_need_columns = [ 'ESG Combined Score 2020',
       'ESG Combined Score 2019', 'ESG Combined Score 2018', 'ESG Score 2020',
       'ESG Score 2019', 'ESG Score 2018', 'ESG Controversy Score 2020',
       'ESG Controversy Score 2019', 'ESG Controversy Score 2018',
       'Environmental Pillar Score 2020', 'Environmental Pillar Score 2019',
       'Environmental Pillar Score 2018',
#       'Workforce Score 2020',
#       'Workforce Score 2019', 'Workforce Score 2018',
       'ESG Innovation Score 2020', 'ESG Innovation Score 2019',
       'ESG Innovation Score 2018', 'R&D/Revenue 2020', 'R&D/Revenue 2019',
       'R&D/Revenue 2018', 'R&D/Employees 2020', 'R&D/Employees 2019',
       'R&D/Employees 2018']

    df = df.drop(columns=not_need_columns)

    for col in df.columns:
        try:
            df[col] = df[col].astype(int)
        except:
            pass

    if win is True:
        df = winsorize_dataframe(df)

    return df
#print(create_labor_sample_data(cleaned_labor_data, sample_sp))

def create_inno_sample_data(sample_df, win=True):
    inno_count = sample_df[['R&D/Employees 2020', 'R&D/Employees 2019', 'R&D/Employees 2018',
                      'R&D/Revenue 2020', 'R&D/Revenue 2019', 'R&D/Revenue 2018']]

    zero_counts = (inno_count == 0).sum()
    inno_mask = (inno_count == 0).any(axis=1)

    df = sample_df[~inno_mask]
    reverse_df = sample_df[inno_mask] # no R&D

    not_need_columns = ['ESG Combined Score 2020',
       'ESG Combined Score 2019', 'ESG Combined Score 2018', 'ESG Score 2020',
       'ESG Score 2019', 'ESG Score 2018', 'ESG Controversy Score 2020',
       'ESG Controversy Score 2019', 'ESG Controversy Score 2018',
       'Environmental Pillar Score 2020', 'Environmental Pillar Score 2019',
       'Environmental Pillar Score 2018', 'Workforce Score 2020',
       'Workforce Score 2019', 'Workforce Score 2018']

    df = df.drop(columns=not_need_columns)
    reverse_df = reverse_df.drop(columns=not_need_columns)

    years = ["2018", "2019", "2020"]
    for year in years:
        df[('R&D/Revenue ' + year)] = (df[('R&D/Revenue ' + year)] - df[('R&D/Revenue ' + year)].min()) / (df[('R&D/Revenue ' + year)].max() - df[('R&D/Revenue ' + year)].min()) * (100 - 0) + 0
        df[('R&D/Employees ' + year)] = (df[('R&D/Employees ' + year)] - df[('R&D/Employees ' + year)].min()) / (df[('R&D/Employees ' + year)].max() - df[('R&D/Employees ' + year)].min()) * (100 - 0) + 0

    if win is True:
        df = winsorize_dataframe(df)

    return df, reverse_df
#inno = create_inno_sample_data(sample_sp)

def create_esg_sample_data(sample_df, win=True):
    not_need_columns = ['Workforce Score 2020',
       'Workforce Score 2019', 'Workforce Score 2018',
       'ESG Innovation Score 2020', 'ESG Innovation Score 2019',
       'ESG Innovation Score 2018', 'R&D/Revenue 2020', 'R&D/Revenue 2019',
       'R&D/Revenue 2018', 'R&D/Employees 2020', 'R&D/Employees 2019',
       'R&D/Employees 2018']
    sample_df = sample_df.drop(columns=not_need_columns)

    if win is True:
        sample_df = winsorize_dataframe(sample_df)

    return sample_df
#esg = create_esg_sample_data(sample_sp)

def create_bias_data(sample_df, win=True):
    not_need_columns = [ 'ESG Combined Score 2020',
       'ESG Combined Score 2019', 'ESG Combined Score 2018', 'ESG Score 2020',
       'ESG Score 2019', 'ESG Score 2018', 'ESG Controversy Score 2020',
       'ESG Controversy Score 2019', 'ESG Controversy Score 2018',
       'Environmental Pillar Score 2020', 'Environmental Pillar Score 2019',
       'Environmental Pillar Score 2018', 'Workforce Score 2020',
       'Workforce Score 2019', 'Workforce Score 2018',
       'ESG Innovation Score 2020', 'ESG Innovation Score 2019',
       'ESG Innovation Score 2018', 'R&D/Revenue 2020', 'R&D/Revenue 2019',
       'R&D/Revenue 2018', 'R&D/Employees 2020', 'R&D/Employees 2019',
       'R&D/Employees 2018']
    sample_df = sample_df.drop(columns=not_need_columns)

    if win is True:
        sample_df = winsorize_dataframe(sample_df)

    return sample_df
#sample_bias = create_bias_data(sample_sp)
#print(sample_bias)


def create_descriptive_data(sample_df, win=True):
    df_descriptive = import_sp_excel_sheet('Quant_MA.xlsx', sheet='Descriptive', sp_ticker_filter=True)

    not_need_columns = [ 'Trade_Union_Rep_2020', 'Trade_Union_Rep_2019', 'Trade_Union_Rep_2018', 'HQ_Country', 'HQ_Region']
    df_descriptive = df_descriptive.drop(columns=not_need_columns)

    df_sample_merge = sample_df[['Ticker', 'GICS Sector']]

    mask = df_descriptive['Ticker'].isin(sample_df['Ticker'])
    df = df_descriptive[mask]

    prefix_mapping = {
        'Common_Name': 'Common Name',
        'Founded': 'Founding Year',
        'RoA_2020': 'Return on Assets 2020',
        'RoA_2019': 'Return on Assets 2019',
        'RoA_2018': 'Return on Assets 2018',
        'BS_2020': 'Book Value per Share 2020',
        'BS_2019': 'Book Value per Share 2019',
        'BS_2018': 'Book Value per Share 2018',
        'Revenue_per_Share_2020': 'Revenue per Share 2020',
        'Revenue_per_Share_2019': 'Revenue per Share 2019',
        'Revenue_per_Share_2018': 'Revenue per Share 2018',
        'PE_Ratio_2020': 'Price/Earnings Ratio 2020',
        'PE_Ratio_2019': 'Price/Earnings Ratio 2019',
        'PE_Ratio_2018': 'Price/Earnings Ratio 2018',
        'Current_Ratio_2020': 'Current Ratio 2020',
        'Current_Ratio_2019': 'Current Ratio 2019',
        'Current_Ratio_2018': 'Current Ratio 2018',
        'TD_to_TA_2020': 'Total Debt/Total Assets 2020',
        'TD_to_TA_2019': 'Total Debt/Total Assets 2019',
        'TD_to_TA_2018': 'Total Debt/Total Assets 2018',
        'EBIT_Margin_2020': 'EBIT Margin 2020',
        'EBIT_Margin_2019': 'EBIT Margin 2019',
        'EBIT_Margin_2018': 'EBIT Margin 2018',
        'Dividend_Payout_Ratio_2020': 'Dividend Payout Ratio 2020',
        'Dividend_Payout_Ratio_2019': 'Dividend Payout Ratio 2019',
        'Dividend_Payout_Ratio_2018': 'Dividend Payout Ratio 2018',
        'Female Board_2020': 'Female Board Percentage 2020',
        'Female Board_2019': 'Female Board Percentage 2019',
        'Female Board_2018': 'Female Board Percentage 2018',
        'Female Executives_2020': 'Female Executives Percentage 2020',
        'Female Executives_2019': 'Female Executives Percentage 2019',
        'Female Executives_2018': 'Female Executives Percentage 2018'
    }

    df.rename(columns=prefix_mapping, inplace=True)

    merged_df = df.merge(df_sample_merge, on='Ticker', how='inner')

    column_order = ['Ticker', 'Common Name', 'GICS Sector', 'Founding Year',
                    'Return on Assets 2020', 'Return on Assets 2019', 'Return on Assets 2018',
                    'Book Value per Share 2020', 'Book Value per Share 2019', 'Book Value per Share 2018',
                    'Revenue per Share 2020', 'Revenue per Share 2019', 'Revenue per Share 2018',
                    'Price/Earnings Ratio 2020', 'Price/Earnings Ratio 2019', 'Price/Earnings Ratio 2018',
                    'Current Ratio 2020', 'Current Ratio 2019', 'Current Ratio 2018',
                    'Total Debt/Total Assets 2020', 'Total Debt/Total Assets 2019', 'Total Debt/Total Assets 2018',
                    'EBIT Margin 2020', 'EBIT Margin 2019', 'EBIT Margin 2018',
                    'Dividend Payout Ratio 2020', 'Dividend Payout Ratio 2019', 'Dividend Payout Ratio 2018',
                    'Female Board Percentage 2020', 'Female Board Percentage 2019', 'Female Board Percentage 2018',
                    'Female Executives Percentage 2020', 'Female Executives Percentage 2019', 'Female Executives Percentage 2018']

    merged_df = merged_df[column_order]

    if win is True:
        merged_df = winsorize_dataframe(merged_df)

    for column in merged_df.select_dtypes(include='object').columns:
        try:
            merged_df[column] = merged_df[column].astype(float)
        except ValueError:
            pass

    return merged_df
#sample_desc = create_descriptive_data(sample_sp)
#print(sample_desc)

# Manuell Pre-Filter (MktCap's empty for any year + Company Dublices
#df = import_ma_excel_sheet(excel_name='Quant_MA.xlsx', sheet='TobinsQ', sp_ticker_filter=False)
#columns_to_check = ["MktCap_2018", "MktCap_2019", "MktCap_2020"]
#nan_mask = df[columns_to_check].applymap(lambda x: isinstance(x, float) and pd.isna(x)).any(axis=1)
#print(df[nan_mask])

# Multiple-Company-Tickers filter
#duplicates = df[df['Common_Name'].duplicated(keep=False)]
#print(duplicates)



# +++++ SENTIMENT ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++ Data Creation (Sentiment) +++
def import_reply_sheet(excel_name, sheet):
    df = pd.read_excel(excel_name, sheet_name=sheet)
    df = df.drop(columns='Unnamed: 0')
    return df

def sentiment_raw(df, sheet):
    df_hiv4 = pd.read_excel('H4_Dict.xlsx', sheet_name='preprocessed')
    df_vader = pd.read_excel('VADER_Dict.xlsx', sheet_name='preprocessed')
    df_swn = pd.read_excel('SWN3.0_Dict.xlsx', sheet_name='SentiWordNet3.0_cleaned')
    df_lm = pd.read_excel('LM_Dict.xlsx', sheet_name='LM')

    sentiment_ma_output = pd.DataFrame(columns=['company_name', 'gunning_fog', 'flesch_kincaid', 'ari',
                                                'h4_pos_score', 'h4_neg_score', 'h4_in_count', 'h4_out_count',
                                                'vader_rating_score', 'vader_in_count', 'vader_out_count',
                                                'swn_pos_score', 'swn_neg_score', 'swn_in_count', 'swn_out_count',
                                                'lm_pos_score', 'lm_neg_score', 'lm_unc_score', 'lm_lit_score',
                                                'lm_str_mod_score',
                                                'lm_wea_mod_score', 'lm_in_count', 'lm_out_count',
                                                'token_numbers', 'string_length'
                                                ])

    for i in range(0, len(df)):
        reply = df['reply'][i].replace(df['company_name'][i], '')
        company_name = df['company_name'][i]
        company_name_split = company_name.split()
        company_name_filtered = [word for word in company_name_split if len(word) > 1]
        pattern = r'\b(?:' + '|'.join(map(re.escape, company_name_filtered)) + r'|' + '|'.join(
            map(lambda word: word + "'s", company_name_filtered)) + r')\b'
        cleaned_reply = re.sub(pattern, ' ', reply)

        cleaned_string_array = [str(sample_sp['Ticker'][i]), 'ConocoPhillips', 'CarMax', 'BorgWarner',
                                'AvalonBday', 'AutoZone', 'ADP', 'AmerisourceBergen', 'plc', 'PLC', 'AbbVie',
                                'www.ebay.com',
                                'DuPont', 'Domino’s', 'McKesson', 'PACCAR', 'PMI', 'PulteGroup', 'ResMed', 'SolarEdge',
                                'KHC',
                                'VeriSign', 'WestRock', 'Corp', 'DexCom', 'FLEERCOR']
        for string in cleaned_string_array:
            cleaned_reply = cleaned_reply.replace(string, '')

        rd_scores = readability_scores(cleaned_reply)
        h4_neg_score, h4_pos_score, h4_str_score, h4_wea_score, h4_act_score, h4_pas_score, h4_count, not_h4_count = calc_h4_dict(
            input=cleaned_reply, df=df_hiv4)
        vader_rating, vader_count, not_vader_count = calc_vader_dict(df=df_vader, input=cleaned_reply)
        swn_neg_score, swn_pos_score, swn_count, not_swn_count = calc_swn_dict(input=cleaned_reply, df=df_swn)
        lm_neg_score, lm_pos_score, lm_unc_score, lm_lit_score, lm_str_mod_score, lm_wea_mod_score, lm_count, not_lm_count = calc_lm_dict(
            input=cleaned_reply, df=df_lm)

        preprocessed_cleaned_reply = preprocess_reply(cleaned_reply)

        cleaned_token_array = ['oration', 'mpany', 'D', 'R', 'O', 'A', 'J', 'B', 'N', 'V']
        preprocessed_cleaned_reply = [token for token in preprocessed_cleaned_reply if token not in cleaned_token_array]

        sentiment_ma_output = sentiment_ma_output.append({
            'company_name': company_name,
            'gunning_fog': rd_scores[0],
            'flesch_kincaid': rd_scores[1],
            'ari': rd_scores[2],
            'h4_pos_score': h4_pos_score,
            'h4_neg_score': h4_neg_score,
            'h4_in_count': h4_count,
            'h4_out_count': not_h4_count,
            'vader_rating_score': vader_rating,
            'vader_in_count': vader_count,
            'vader_out_count': not_vader_count,
            'swn_pos_score': swn_pos_score,
            'swn_neg_score': swn_neg_score,
            'swn_in_count': swn_count,
            'swn_out_count': not_swn_count,
            'lm_pos_score': lm_pos_score,
            'lm_neg_score': lm_neg_score,
            'lm_unc_score': lm_unc_score,
            'lm_lit_score': lm_lit_score,
            'lm_str_mod_score': lm_str_mod_score,
            'lm_wea_mod_score': lm_wea_mod_score,
            'lm_in_count': lm_count,
            'lm_out_count': not_lm_count,
            'token_numbers': len(preprocessed_cleaned_reply),
            'string_length': len(df['reply'][i])

        }, ignore_index=True)

        print(i)

    with pd.ExcelWriter("Sentiment_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        sentiment_ma_output.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0)

    #return sentiment_ma_output

def scale_sentiment_(sheet):
    df = pd.read_excel("Sentiment_MA.xlsx", sheet_name=sheet)
    df = df.drop(columns='Unnamed: 0')

    write_to = sheet.replace('_raw','')
    print(write_to)

    columns_to_divide = ['h4_pos_score', 'h4_neg_score', 'h4_in_count', 'h4_out_count',
                         'vader_rating_score', 'vader_in_count', 'vader_out_count',
                         'swn_pos_score', 'swn_neg_score', 'swn_in_count', 'swn_out_count',
                         'lm_pos_score', 'lm_neg_score', 'lm_unc_score', 'lm_lit_score',
                         'lm_str_mod_score', 'lm_wea_mod_score', 'lm_in_count', 'lm_out_count']

    df[columns_to_divide] = df[columns_to_divide].div(df['token_numbers'], axis=0)

    with pd.ExcelWriter("Sentiment_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=write_to, startrow=0, startcol=0)

    return df

#replys = import_reply_sheet(excel_name='Responses_MA.xlsx', sheet='Labor_2020')
#sentiment_raw =sentiment_raw(df=replys, sheet="Labor_2020_raw")
#sentiment_scaled = scale_sentiment_(sheet="Labor_2020_raw")

#replys = import_reply_sheet(excel_name='Responses_MA.xlsx', sheet='Performance')
#sentiment_raw =sentiment_raw(df=replys, sheet="Performance_raw")
#sentiment_scaled = scale_sentiment_(sheet="Performance_raw")

# manually create the Sentiment for Business_Summary by REFINITIV
"""
df_hiv4 = pd.read_excel('H4_Dict.xlsx', sheet_name='preprocessed')
df_vader = pd.read_excel('VADER_Dict.xlsx', sheet_name='preprocessed')
df_swn = pd.read_excel('SWN3.0_Dict.xlsx', sheet_name='SentiWordNet3.0_cleaned')
df_lm = pd.read_excel('LM_Dict.xlsx', sheet_name='LM')

sentiment_ma_output = pd.DataFrame(columns=['company_name', 'gunning_fog', 'flesch_kincaid', 'ari',
                           'h4_pos_score', 'h4_neg_score', 'h4_in_count', 'h4_out_count',
                           'vader_rating_score', 'vader_in_count', 'vader_out_count',
                           'swn_pos_score', 'swn_neg_score', 'swn_in_count', 'swn_out_count',
                           'lm_pos_score', 'lm_neg_score', 'lm_unc_score', 'lm_lit_score', 'lm_str_mod_score',
                           'lm_wea_mod_score', 'lm_in_count', 'lm_out_count',
                           'token_numbers', 'string_length' 
                           ])

for i in range(0, len(sp_aea_data)):
    reply = sp_aea_data['Business_Summary'][i].replace(sp_aea_data['Common_Name'][i], '')
    company_name = sp_aea_data['Common_Name'][i]
    company_name_split = company_name.split()
    company_name_filtered = [word for word in company_name_split if len(word) > 1]
    pattern = r'\b(?:' + '|'.join(map(re.escape, company_name_filtered)) + r'|' + '|'.join(
        map(lambda word: word + "'s", company_name_filtered)) + r')\b'
    cleaned_reply = re.sub(pattern, ' ', reply)

    cleaned_string_array = [str(sp_aea_data['Ticker'][i]), 'ConocoPhillips', 'CarMax', 'BorgWarner',
                     'AvalonBday', 'AutoZone', 'ADP', 'AmerisourceBergen', 'plc', 'PLC', 'AbbVie', 'www.ebay.com',
                     'DuPont', 'Domino’s', 'McKesson', 'PACCAR', 'PMI', 'PulteGroup', 'ResMed', 'SolarEdge', 'KHC',
                     'VeriSign', 'WestRock', 'Corp', 'DexCom', 'FLEERCOR']
    for string in cleaned_string_array:
        cleaned_reply = cleaned_reply.replace(string, '')

    rd_scores = readability_scores(cleaned_reply)
    h4_neg_score, h4_pos_score, h4_str_score, h4_wea_score, h4_act_score, h4_pas_score, h4_count, not_h4_count = calc_h4_dict(
        input=cleaned_reply, df=df_hiv4)
    vader_rating, vader_count, not_vader_count = calc_vader_dict(df=df_vader, input=cleaned_reply)
    swn_neg_score, swn_pos_score, swn_count, not_swn_count = calc_swn_dict(input=cleaned_reply, df=df_swn)
    lm_neg_score, lm_pos_score, lm_unc_score, lm_lit_score, lm_str_mod_score, lm_wea_mod_score, lm_count, not_lm_count = calc_lm_dict(
        input=cleaned_reply, df=df_lm)

    preprocessed_cleaned_reply = preprocess_reply(cleaned_reply)

    cleaned_token_array = ['oration', 'mpany', 'D', 'R', 'O', 'A', 'J', 'B', 'N', 'V']
    preprocessed_cleaned_reply = [token for token in preprocessed_cleaned_reply if token not in cleaned_token_array]

    sentiment_ma_output = sentiment_ma_output.append({
        'company_name': company_name,
        'gunning_fog': rd_scores[0],
        'flesch_kincaid': rd_scores[1],
        'ari': rd_scores[2],
        'h4_pos_score': h4_pos_score,
        'h4_neg_score': h4_neg_score,
        'h4_in_count': h4_count,
        'h4_out_count': not_h4_count,
        'vader_rating_score': vader_rating,
        'vader_in_count': vader_count,
        'vader_out_count': not_vader_count,
        'swn_pos_score': swn_pos_score,
        'swn_neg_score': swn_neg_score,
        'swn_in_count': swn_count,
        'swn_out_count': not_swn_count,
        'lm_pos_score': lm_pos_score,
        'lm_neg_score': lm_neg_score,
        'lm_unc_score': lm_unc_score,
        'lm_lit_score': lm_lit_score,
        'lm_str_mod_score': lm_str_mod_score,
        'lm_wea_mod_score': lm_wea_mod_score,
        'lm_in_count': lm_count,
        'lm_out_count': not_lm_count,
        'token_numbers': len(preprocessed_cleaned_reply),
        'string_length': len(sp_aea_data['Business_Summary'][i])

    }, ignore_index=True)

    print(i)

with pd.ExcelWriter("Sentiment_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    sentiment_ma_output.to_excel(writer, sheet_name='Overview_Refinitiv_raw', startrow=0, startcol=0)
"""

# +++ Data Analysis (Sentiment) +++
def import_sentiment_sheet(sheet, year, drop_lm=True, corr=False, adapt=False, adapt_df=None):
    if year is None:
        df = pd.read_excel('Sentiment_MA.xlsx', sheet_name=sheet)
        new_year = ''
    else:
        new_year = ('_' + year)
        df = pd.read_excel('Sentiment_MA.xlsx', sheet_name=(sheet + new_year))
        new_year = (' ' + year)
    if sheet == 'Overview':
        new_year = ' ChatGPT'
    elif sheet == 'Overview_Refinitiv':
        new_year = ' Refinitiv'
    df = df.drop(columns='Unnamed: 0')
    prefix_mapping = {
        'company_name': 'Common Name',
        'gunning_fog': ('Gunning Fog' + new_year),
        'flesch_kincaid': ('Flesch Kincaid' + new_year),
        'ari': ('ARI' + new_year),
        'h4_pos_score': ('H4 Positive' + new_year),
        'h4_neg_score': ('H4 Negative' + new_year),
        'h4_in_count': ('In H4' + new_year),
        'h4_out_count': ('Out H4' + new_year),
        'vader_rating_score': ('Vader Rating' + new_year),
        'vader_in_count': ('In Vader' + new_year),
        'vader_out_count': ('Out Vader' + new_year),
        'swn_pos_score': ('SWN Positive' + new_year),
        'swn_neg_score': ('SWN Negative' + new_year),
        'swn_in_count': ('In SWN' + new_year),
        'swn_out_count': ('Out SWN' + new_year),
        'lm_pos_score': ('LM Positive' + new_year),
        'lm_neg_score': ('LM Negative' + new_year),
        'lm_unc_score': ('LM Unc Score'),
        'lm_lit_score': ('LM Lit Score'),
        'lm_str_mod_score': ('LM Str Modal Score'),
        'lm_wea_mod_score': ('LM Wea Modal Score'),
        'lm_in_count': ('In LM' + new_year),
        'lm_out_count': ('Out LM' + new_year),
        'token_numbers': ('Number of Tokens' + new_year),
        'string_length': ('Length of String' + new_year)
    }

    df.rename(columns=prefix_mapping, inplace=True)

    df = df.set_index('Common Name')

    if drop_lm is True:
        df = df.drop(columns=['LM Unc Score', 'LM Lit Score', 'LM Str Modal Score', 'LM Wea Modal Score',
                              'Out LM' + new_year, 'Out SWN' + new_year, 'Out Vader' + new_year, 'Out H4' + new_year,#])
                              'Length of String' + new_year]) # take this out for correlations
    if corr is True:
        df = df[['LM Positive' + new_year,'LM Negative' + new_year, 'Number of Tokens' + new_year]]

    if adapt is True:
        df = df.loc[df.index.isin(adapt_df['Common Name'])]

    return df

def import_sentiment_sheet_raw(excel_name, sheet):
    df = pd.read_excel(excel_name, sheet_name=sheet)
    df = df.drop(columns='Unnamed: 0')
    return df # # #

def sent_perc(df, dict, sent, quant):
    choices = ['LM Sentiment ChatGPT', 'LM Sentiment Refinitiv', 'LM Sentiment 2018', 'LM Sentiment 2019', 'LM Sentiment 2020']
    for var in choices:
        try:
            if dict == 'LM':
                if sent == 'pos':
                    var = var.replace('Sentiment', 'Positive')
                elif sent == 'neg':
                    var = var.replace('Sentiment', 'Negative')
            percentile_value = df[var].quantile(quant)
            if quant == 0.9:
                filtered_df = df[var] >= percentile_value
            elif quant == 0.1:
                filtered_df = df[var] <= percentile_value
            return df.loc[filtered_df][var]
        except KeyError:
            continue  # Move to the next variable if the current one raises a KeyError
    raise ValueError("None of the choices worked without error.")

def sent_perc_best_worst_sentiment(df):
    df_most_pos = sent_perc(df, dict='LM', sent='pos', quant=0.9)
    print("Amount of Companies with a scaled positive Word-Count above the 90% quantile (scaled positive Word-Count):", len(df_most_pos), ', equal to:', round((len(df_most_pos)/len(df))*100,4), '%')
    average_value = df_most_pos.mean()
    print("Average scaled positive Word-Count for the Companies above the 90% quantile (scaled positive Word-Count):", (round(average_value,4)))


    df_least_pos = sent_perc(df, dict='LM', sent='pos', quant=0.1)
    print("Amount of Companies with a scaled positive Word-Count below the 10% quantile (scaled positive Word-Count)", len(df_least_pos), ', equal to:', round((len(df_least_pos)/len(df))*100,4), '%')
    average_value = df_least_pos.mean()
    print("Average scaled positive Word-Count for the Companies below the 10% quantile (scaled positive Word-Count):", (round(average_value,4)))


    df_most_neg = sent_perc(df, dict='LM', sent='neg', quant=0.9)
    print("Amount of Companies with a scaled negative Word-Count above the 90% quantile (scaled negative Word-Count):", len(df_most_neg), ', equal to:', round((len(df_most_neg)/len(df))*100,4), '%')
    average_value = df_most_neg.mean()
    print("Average scaled negative Word-Count for the Companies above the 90% quantile (scaled negative Word-Count):", (round(average_value,4)))

    df_least_neg = sent_perc(df, dict='LM', sent='neg', quant=0.1)
    print("Amount of Companies with a scaled negative Word-Count below the 10% quantile (scaled negative Word-Count):", len(df_least_neg), ', equal to:', round((len(df_least_neg)/len(df))*100,4), '%')
    average_value = df_least_neg.mean()
    print("Average scaled negative Word-Count for the Companies below the 10% quantile (scaled negative Word-Count):", (round(average_value,4)))

    best_df = pd.merge(df_most_pos, df_least_neg, left_index=True, right_index=True, how='inner')
    best_df_len = len(best_df)
    best_df_perc = (best_df_len / len(df)) * 100
    print(f"Percentage of Companies with a scaled positive Word-Count above the 90% Percentile and have a negative Word-Count below the 10% Percentile: {best_df_len} or {best_df_perc:.4f}%")
    average_value_pos = best_df.iloc[:,0].mean()
    average_value_neg = best_df.iloc[:,1].mean()
    print(f"Average scaled positive (negative) Word-Count for the best Companies: {average_value_pos:.4f} ({average_value_neg:.4f})")


    worst_df = pd.merge(df_least_pos, df_most_neg, left_index=True, right_index=True, how='inner')
    worst_df_len = len(worst_df)
    worst_df_perc = (worst_df_len / len(df)) * 100
    print(f"Percentage of Companies with a scaled positive Word-Count below the 10% Percentile and have a negative Word-Count above the 90% Percentile: {worst_df_len} or {worst_df_perc:.4f}%")
    average_value_pos = worst_df.iloc[:,0].mean()
    average_value_neg = worst_df.iloc[:,1].mean()
    print(f"Average scaled positive (negative) Word-Count for the best Companies: {average_value_pos:.4f} ({average_value_neg:.4f})")

    return best_df, worst_df

def count_sector_and_names_best_worst_sent(sample_df, best_worst_df, sheet_title):
    merged_best = merge_common_index(sample_df, best_worst_df[0])
    merged_worst = merge_common_index(sample_df, best_worst_df[1])
    merged_best = merged_best[['Common Name', 'GICS Sector', 'GICS Industry']]
    merged_worst = merged_worst[['Common Name', 'GICS Sector', 'GICS Industry']]
    with pd.ExcelWriter("BWCount_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        merged_best.to_excel(writer, sheet_name='sent_best_' + sheet_title, startrow=0, startcol=0)
    with pd.ExcelWriter("BWCount_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        merged_worst.to_excel(writer, sheet_name='sent_worst_' + sheet_title, startrow=0, startcol=0)

def plot_sector_distribution_best_worst_lm_sent(sample_df, best_worst_df, over_title):
    merged_best = merge_common_index(sample_df, best_worst_df[0])
    merged_worst = merge_common_index(sample_df, best_worst_df[1])

    font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
    font_prop = font_manager.FontProperties(fname=font_path, size=12)

    def generate_subplot(data_frame, ax, title):
        gics_sector_counts = data_frame['GICS Sector'].value_counts(normalize=True) * 100
        font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
        font_prop = font_manager.FontProperties(fname=font_path, size=12)
        ax = gics_sector_counts.plot(kind='bar', color='grey', width=0.55, ax=ax)
        ax.set_title(title, fontproperties=font_prop, weight='bold', size=18)
#        ax.set_xlabel('Sector', fontproperties=font_prop, weight='bold', size=15)
#        ax.set_ylabel('Percentage of Companies', fontproperties=font_prop, weight='bold', size=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right', fontproperties=font_prop)
        ax.set_yticklabels([f'{val:.2f}%' for val in ax.get_yticks()], fontproperties=font_prop, size=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7.5), textcoords='offset points',
                        fontproperties=font_prop, size=10)
        source_text = 'Source: Refinitiv, 2023'
        plt.figtext(0.99, 0.01, source_text, ha='right', fontproperties=font_prop)

    fig, axes = plt.subplots(1, 2, figsize=(24, 13))
    generate_subplot(merged_best, axes[0], 'Best LM-Dictionary Sentiment, n =' + str(len(merged_best)))
    generate_subplot(merged_worst, axes[1], 'Worst LM-Dictionary Sentiment, n =' + str(len(merged_worst)))

    plt.suptitle(over_title, fontsize=20, fontweight='bold', fontproperties=font_prop)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing and position for overall title
    plt.show()



# +++++ GRADES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++ Data Creation (Grades) +++

def import_grades_sheet(sheet, year):
    if year is None:
        df = pd.read_excel('Grades_MA.xlsx', sheet_name=sheet)
        new_year = ''
    else:
        new_year = ('_' + year)
        df = pd.read_excel('Grades_MA.xlsx', sheet_name=(sheet + new_year))

    df = df.drop(columns='Unnamed: 0')

    prefix_mapping = {
        'company_name': 'Common Name'}

    df.rename(columns=prefix_mapping, inplace=True)

    df = df.set_index('Common Name')

    df['Grade Score'] = df['Grade'].apply(lambda x: grade_to_score(x))

    if sheet == 'ESG':
        df[('Grade Score ESG ' + year)] = df['Grade Score']
    elif sheet == 'Innovation':
        df[('Grade Score Innovation ' + year)] = df['Grade Score']
    elif sheet == 'Labor':
        df[('Grade Score Labor ' + year)] = df['Grade Score']
    else:
        df[('Grade Score ' + sheet)] = df['Grade Score']

    df = df.drop(columns=['Grade', 'Grade Score', 'reply'])

    return df

def get_grade(sheet_name):
    sheet = import_reply_sheet(excel_name='Responses_MA.xlsx', sheet=sheet_name)
    sheet["Grade"] = sheet["reply"].str.extract(r'Grade: ([A-Z+-]+)$')
    sheet = sheet[['company_name', 'Grade', 'reply']]
    with pd.ExcelWriter("Grades_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        sheet.to_excel(writer, sheet_name=('Grade_'+ sheet_name), startrow=0, startcol=0)
#Innovation_2020 = get_grade(sheet_name='Innovation_2020')

def grade_sample_merge(df_sample, df_grade, ss, rq, year):
    df = df_sample.merge(df_grade, left_on='Common Name', right_index=True)
    if ss is True:
        df = df_sample.merge(df_grade, left_on='Common Name', right_index=True).drop(columns=['Common Name', 'Size ' + year, 'Goodwill ' + year, 'SGA ' + year,
                                                                                              'Donations/Revenue ' + year, 'GSV Names ' + year, 'GSV Tickers ' + year])
    if rq == 'ESG':
        df[('Grade Score Deviation ESG (abs.) ' + year)] = abs(df['ESG Score ' + year] - df[('Grade Score ESG ' + year)])
        df[('Grade Score Deviation ESG ' + year)] = (df['ESG Score ' + year] - df[('Grade Score ESG ' + year)])

    elif rq == 'Labor':
        df[('Grade Score Deviation Labor (abs.) ' + year)] = abs(df['Employee Turnover Score ' + year] - df[('Grade Score Labor ' + year)])
        df[('Grade Score Deviation Labor ' + year)] = (df['Employee Turnover Score ' + year] - df[('Grade Score Labor ' + year)])

    elif rq == 'Innovation':
        df[('Grade Score Deviation Innovation (abs.) ' + year)] = abs(df[('R&D/Revenue ' + year)] - df[('Grade Score Innovation ' + year)])
        df[('Grade Score Deviation Innovation ' + year)] = (df[('R&D/Revenue ' + year)] - df[('Grade Score Innovation ' + year)])

    elif rq == 'Performance':
        df[('Return Deviation (vs. Past Year)')] = (df['1y Past Return (Refinitiv)'] - df['Average Upc. Return (ChatGPT)'])
        df[('Return Deviation (vs. Upc. Year)')] = (df['1y Upc. Return (Refinitiv)'] - df['Average Upc. Return (ChatGPT)'])
        df = df.drop(columns=['company_name', 'reply', '10y Past Return (Refinitiv)', '5y Past Return (Refinitiv)', '3y Past Return (Refinitiv)'])
    return df

def grade_sample_sent_merge(df_sample, df_grade, df_sent, rq ,year):
    df_sample = df_sample.drop(columns=['Size ' + year, 'Goodwill ' + year, 'SGA ' + year,
                                        'Donations/Revenue ' + year,
                                        'GSV Names ' + year, 'GSV Tickers ' + year]).set_index('Common Name')

    df = df_sample.merge(df_grade, left_on='Common Name', right_index=True)

    if rq == 'ESG':
        df[('Grade Score Deviation ESG (abs.) ' + year)] = abs(df['ESG Score ' + year] - df[('Grade Score ESG ' + year)])
        df[('Grade Score Deviation ESG ' + year)] = (df['ESG Score ' + year] - df[('Grade Score ESG ' + year)])

        df = df.merge(df_sent, left_on='Common Name', right_index=True).drop(columns=['ESG Combined Score ' + year,
                                                                                      'ESG Controversy Score ' + year,
                                                                                      'Environmental Pillar Score ' + year])
    elif rq == 'Labor':
        df[('Grade Score Deviation Labor (abs.) ' + year)] = abs(df['Employee Turnover Score ' + year] - df[('Grade Score Labor ' + year)])
        df[('Grade Score Deviation Labor ' + year)] = (df['Employee Turnover Score ' + year] - df[('Grade Score Labor ' + year)])

        df = df.merge(df_sent, left_on='Common Name', right_index=True).drop(columns=['Workforce Score ' + year])

    elif rq == 'Innovation':
        df[('R&D/Revenue (scaled) ' + year)] = (df[('R&D/Revenue ' + year)] - df[('R&D/Revenue ' + year)].min()) / (df[('R&D/Revenue ' + year)].max() - df[('R&D/Revenue ' + year)].min()) * (100 - 0) + 0
        df[('Grade Score Deviation Innovation (abs.) ' + year)] = abs(df[('R&D/Revenue (scaled) ' + year)] - df[('Grade Score Innovation ' + year)])
        df[('Grade Score Deviation Innovation ' + year)] = (df[('R&D/Revenue (scaled) ' + year)] - df[('Grade Score Innovation ' + year)])

        df = df.merge(df_sent, left_on='Common Name', right_index=True).drop(columns=[('R&D/Revenue ' + year), ('R&D/Employees ' + year), ('ESG Innovation Score ' + year)])

    return df
    # +++ Data Analysis (Grades) +++

def grade_to_score(grade):
    grade_ranges = {
        'D-': (0.0, 0.083333),
        'D': (0.083333, 0.166666),
        'D+': (0.166666, 0.250000),
        'C-': (0.250000, 0.333333),
        'C': (0.333333, 0.416666),
        'C+': (0.416666, 0.500000),
        'B-': (0.500000, 0.583333),
        'B': (0.583333, 0.666666),
        'B+': (0.666666, 0.750000),
        'A-': (0.750000, 0.833333),
        'A': (0.833333, 0.916666),
        'A+': (0.916666, 1.0)
    }
    for grade_key, grade_range in grade_ranges.items():
        if grade_key == grade:
            return (grade_range[0] + grade_range[1]) / 2 * 100
    return None

def grades_perc_best_worst(df, df_desc, df_sample, rq, year):

    if rq == 'ESG':
        column = 'Grade Score Deviation ESG ' + year
    elif rq == 'Innovation':
        column = 'Grade Score Deviation Innovation ' + year
    elif rq == 'Labor':
        column = 'Grade Score Deviation Labor ' + year
    elif rq == 'Performance':
        column = 'Return Deviation (vs. Upc. Year)'

    df = df[column]
    upper_percentile = df.quantile(0.9)
    upper_df = df[df >= upper_percentile]
    upper_df_len = (len(upper_df) / len(df)) * 100
    print(f"Percentage of Companies with a Grade chosen by ChatGPT above the 90% Percentile: {len(upper_df)} or {upper_df_len:.4f}%")
    average_value = upper_df.mean()
    print("Average Grade for the Companies above the 90% Percentile:", (round(average_value,4)))


    lower_percentile = df.quantile(0.1)
    lower_df = df[df <= lower_percentile]
    lower_df_len = (len(lower_df) / len(df)) * 100
    print(f"Percentage of Companies with a Grade chosen by ChatGPT below the 10% Percentile: {len(lower_df)} or {lower_df_len:.4f}%")
    average_value = lower_df.mean()
    print("Average Grade for the Companies below the 10% Percentile:", (round(average_value,4)))

    excel_columns = ['Common Name', 'GICS Sector', 'GICS Industry']
    df_excel = df_sample[excel_columns]
    if rq == 'Performance':
        df_excel = df_excel.set_index('Common Name')

    upper_df_excel = df_excel.merge(upper_df.to_frame(), left_index=True, right_index=True)
    with pd.ExcelWriter("BWCount_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        upper_df_excel[['Common Name', 'GICS Sector', 'GICS Industry']].to_excel(writer, sheet_name='grad_worst_' + rq + '_' + year, startrow=0, startcol=0)
    lower_df_excel = df_excel.merge(lower_df.to_frame(), left_index=True, right_index=True)
    with pd.ExcelWriter("BWCount_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        lower_df_excel[['Common Name', 'GICS Sector', 'GICS Industry']].to_excel(writer, sheet_name='grad_best_' + rq + '_' + year, startrow=0, startcol=0)
        #df_desc.drop(columns=[excel_columns]).
    df_desc['Founding Year ' + year] = df_desc['Founding Year']
    df_desc = df_desc.loc[:,df_desc.columns.str.contains(year) | (df_desc.columns == 'Common Name')]
    upper_df = upper_df_excel.merge(df_desc, on="Common Name", how="inner").drop(columns=['Common Name', 'GICS Sector', 'GICS Industry', column])
    lower_df = lower_df_excel.merge(df_desc, on="Common Name", how="inner").drop(columns=['Common Name', 'GICS Sector', 'GICS Industry', column])

    return upper_df, lower_df

def import_performance_sheet():
    df = pd.read_excel("Responses_MA.xlsx", sheet_name='Performance')
    df = df.drop(columns='Unnamed: 0')
    df[['Common Name','Upper', 'Lower']] = df[['company_name','upper', 'lower']]
    df['Range'] = df['upper'] - df['lower']
    df['Mid'] = (df['upper'] + df['lower']) / 2
    df = df[['Common Name', 'Range', 'Mid']].set_index('Common Name')
    return df
#performance_sheet = import_performance_sheet()
#print(performance_sheet)


# +++++ Graphics & Statistics ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_distribution(df, column_name): # 315x130 (Screenshot)
    font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
    font_prop = font_manager.FontProperties(fname=font_path, size=12)

    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    plt.figure(figsize=(24, 12))#, tight_layout=True)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    sns.histplot(data=df, x=column_name, kde=True, bins=50, color='grey', alpha=0.7)
    plt.xlabel(column_name, fontproperties=font_prop, weight='bold', size=15)
    plt.title(f'Distribution of {column_name}', fontproperties=font_prop, weight='bold', size=18)
    plt.ylabel('Frequency', fontproperties=font_prop, weight='bold', size=15)
    source_text = 'Source: Refinitiv, 2023'
    plt.figtext(0.98, 0.02, source_text, ha='right', fontproperties=font_prop)
    plt.show()
#plot_distribution(var_ma_data, 'SGA 2019', plot_type='hist')

def plot_table(df, title, source):
    df_formatted = df.applymap(lambda x: f'{x:.2f}' if pd.notnull(x) and isinstance(x, (int, float)) else x)

    styled_df = df_formatted.style\
        .set_table_styles([{'selector': 'tr:hover', 'props': 'background-color: yellow;'},
                           {'selector': 'th', 'props': [('border-right', '1px solid black'),
                                                        ('border-bottom', '1px solid black'),
                                                        ('padding', '5px'),
                                                        ('font-weight', 'bold'),
                                                        ('font-style', 'italic'),
                                                        ('class', 'header-row')]},  # Right border and bold text for header cells
                           {'selector': 'td', 'props': [('border-right', '1px solid black'),
                                                        ('border-bottom', '1px solid black'),
                                                        ('padding', '3px'),
                                                        ('text-align', 'center'),
                                                        ('vertical-align', 'middle'),
                                                        ('width', '80px'),
                                                        ('height', '25px')]},  # Set a fixed height for rows
                           {'selector': 'th:first-child', 'props': [('border-left', 'none')]},
                           {'selector': 'td:first-child', 'props': [('border-left', 'none')]},
                           {'selector': 'th:last-child', 'props': [('border-right', 'none')]},
                           {'selector': 'td:last-child', 'props': [('border-right', 'none')]},
                           {'selector': 'th:nth-child(n+2):nth-child(-n+3)', 'props': [('font-weight', 'bold')]}
                           ])\
        .set_properties(**{'font-family': 'Times New Roman', 'font-size': '12pt'})\
        .set_table_attributes('style="border-collapse: collapse;"')

    table_html = styled_df.render()

    html_str = f"<div style='text-align: center;'>"
    html_str += "<div style='margin: auto; display: inline-block;'>"

    if title:
        html_str += f"<h2>{title}</h2>"
    html_str += table_html
    html_str += "</div>"

    if source:
        html_str += f"<p>{source}</p>"
    html_str += "</div>"

    return html_str

def calculate_descriptive_statistics(df, year, drop=False):
    if year is not None:
        try:
            df['Founding Year ' + year] = df['Founding Year']
            filtered_data = df.loc[:, df.columns.str.contains(year)]
        except:
            filtered_data = df.loc[:, df.columns.str.contains(year)]

    else:
        filtered_data = df

    summary_stats_df = pd.DataFrame(columns=['Mean', 'SD', 'Skew', 'Kurt', 'Min', '0.25', 'Med', '0.75', 'Max', 'amount of 0s'])

    if drop is True:
        filtered_data = filtered_data.drop(columns=['Goodwill ' + year,
                              'SGA '+ year,
                              'Size '+ year,
                              'Donations/Revenue ' + year,
                              'GSV Names ' + year,
                              'GSV Tickers ' + year])

    for column in filtered_data.columns:
        mean = filtered_data[column].mean()
        std_dev = filtered_data[column].std()
        skewness = filtered_data[column].skew()
        kurtosis = filtered_data[column].kurtosis()
        min_value = filtered_data[column].min()
        quartile_25 = filtered_data[column].quantile(0.25)
        median = filtered_data[column].median()
        quartile_75 = filtered_data[column].quantile(0.75)
        max_value = filtered_data[column].max()
        num_zeros = (filtered_data[column] == 0).sum()

        summary_stats_df = summary_stats_df.append({
            'Mean': mean,
            'SD': std_dev,
            'Skew': skewness,
            'Kurt': kurtosis,
            'Min': min_value,
            '0.25': quartile_25,
            'Med': median,
            '0.75': quartile_75,
            'Max': max_value,
            'amount of 0s': num_zeros,
        }, ignore_index=True)

    summary_stats_df = summary_stats_df.round(2)
    summary_stats_df.index = filtered_data.columns
#    summary_stats_df = summary_stats_df.set_index('Column')

    return summary_stats_df
#quant_data_ds = calculate_descriptive_statistics(quant_data, year='2018')

def calculate_correlation_heatmap(df, title, source, tight=False):
    corr = df.corr()
    font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
    font_prop = font_manager.FontProperties(fname=font_path, size=12)
    source_text = source
    plt.figure(figsize=(20, 12), tight_layout=tight)
    #plt.figure(figsize=(18, 10), tight_layout=tight)
    #plt.title(title, fontproperties=font_prop,
    #          weight='bold', size=18)
    lower = np.triu(corr)
    annot_kws = {'size': 9, 'fontproperties': font_prop}
    ax = sns.heatmap(corr, annot=True, fmt=".2f", linewidth=0.1, cbar=False, mask=lower, annot_kws=annot_kws)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, horizontalalignment='right', size=9, fontproperties=font_prop)
    ax.set_yticklabels(ax.get_yticklabels(), size=9, fontproperties=font_prop)
    if tight is True:
        plt.figtext(0.99, 0.01, source_text, ha='right', fontproperties=font_prop, fontsize=9)
    else:
        plt.figtext(0.98, 0.02, source_text, ha='right', fontproperties=font_prop, fontsize=9)
    plt.show()

def plot_cloud(df_quant, df_qual, year, rq, sent_type='in', title=None):
    font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
    font_prop = font_manager.FontProperties(fname=font_path, size=12)

    if year is not None:
        df_quant = df_quant.loc[:, df_quant.columns.str.contains(year)]
        new_column_names = []
        for col in df_quant.columns:
            new_column_names.append(col.replace((' ' + year), ''))
        df_quant.columns = new_column_names
        df_quant = df_quant[rq]
        if sent_type == 'in':
            y_label = 'Percentage of Words in Dictionary'
            df_qual = df_qual[['In H4 Percentage', 'In SWN Percentage', 'In LM Percentage', 'In Vader Percentage']]
        elif sent_type == 'neg':
            y_label = 'Negative Word-Counts (scaled)'
            df_qual = df_qual[['H4 Negative Score', 'SWN Negative Score', 'LM Negative Score']]
        elif sent_type == 'pos':
            y_label = 'Positive Word-Counts (scaled)'
            df_qual = df_qual[['H4 Positive Score', 'SWN Positive Score', 'LM Positive Score']]
        elif sent_type == 'len':
            y_label = 'Volume of ChatGPT´s Reply'
            df_qual = df_qual[['Number of Tokens', 'Length of String']]
    else:
        None

    plt.figure(figsize=(14, 8))
    markers = ['o', 'X', 'D', 's']
    colors = ['indianred', 'forestgreen', 'royalblue', 'gold']

    for i, col in enumerate(df_qual.columns):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        plt.scatter(df_quant, df_qual[col], label=col, marker=marker, s=10, color=color)

    plt.xlabel(rq, fontproperties=font_prop, weight='bold', size=15)
    plt.ylabel(y_label, fontproperties=font_prop, weight='bold', size=15)
    plt.title(title, fontproperties=font_prop, weight='bold', size=18)
    plt.xticks(fontproperties=font_prop, size=10)
    plt.yticks(fontproperties=font_prop, size=10)
    plt.legend(loc='upper right', prop=font_prop)
    plt.show()
    return None
#title_name = 'Descriptive Statistics: Sentiment for Overview prompt'
#plot_df = plot_cloud(df_quant=quant_data, df_qual=sample_sp, year='2018', rq='ESG Score', sent_type='pos', title=title_name)
#print(plot_df)

def import_all_bw():
    types = ["sent", "grad"]
    bws = ["best", "worst"]
    prompts = ["overview", "ESG", "Labor", "Innovation", "Performance"]
    years = [2018, 2019, 2020]
    dfs = []
    for typ in types:
        for bw in bws:
            for prompt in prompts:
                if prompt == "overview":
                    if typ == "grad":
                        continue
                    sheet = typ + "_" + bw + "_" + prompt
                    df = pd.read_excel("BWCount_MA.xlsx", sheet_name=sheet)
                    df["Prompt"] = sheet
                    print(sheet)
                    dfs.append(df)
                elif prompt == "Performance":
                    sheet = typ + "_" + bw + "_" + prompt
                    df = pd.read_excel("BWCount_MA.xlsx", sheet_name=sheet)
                    df["Prompt"] = sheet
                    print(sheet)
                    dfs.append(df)
                else:
                    for year in years:
                        #try:
                            sheet = typ + "_" + bw + "_" + prompt + "_" + str(year)
                            df = pd.read_excel("BWCount_MA.xlsx", sheet_name=sheet)
                            df["Prompt"] = sheet
                            print(sheet)
                            dfs.append(df)
                    #except:
                    #    None

    result_df = pd.concat(dfs, ignore_index=True)
    #result_df = result_df.dr<<yop(columns='Unnamed: 0 ')

    with pd.ExcelWriter("Further Findings.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name='BW_raw', startrow=0, startcol=0)

#print(import_all_bw())

# +++++ DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Summary/Descriptive Statistics +++
# + BIAS Measurements +
"""
Descriptive Statistics: Influencing  Variables Years 2018-2020
"""
#sample_bias = create_bias_data(sample_sp)
#ds_sample_bias_18 = calculate_descriptive_statistics(sample_bias, year='2018')
#ds_sample_bias_19 = calculate_descriptive_statistics(sample_bias, year='2019')
#ds_sample_bias_20 = calculate_descriptive_statistics(sample_bias, year='2020')
#merged_df = pd.concat([ds_sample_bias_18, empty_df_one, ds_sample_bias_19, empty_df_two, ds_sample_bias_20], ignore_index=False)
# + DESC Measurements +
"""
Descriptive Statistics: Fundamental Variables Years 2018-2020
"""
#sample_desc = create_descriptive_data(sample_sp)
#ds_sample_desc_18 = calculate_descriptive_statistics(sample_desc, year='2018')
#ds_sample_desc_19 = calculate_descriptive_statistics(sample_desc, year='2019')
#ds_sample_desc_20 = calculate_descriptive_statistics(sample_desc, year='2020')

# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sample_bias_18
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sample_desc_18, empty_df_one, ds_sample_desc_19, empty_df_two, ds_sample_desc_20], ignore_index=False)
#merged_df = pd.concat([ds_18, empty_df_one, ds_19], ignore_index=False)

# plot the merged-df and create .hmtl file
##table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)


# +++ Correlation-Heatmaps +++
# + BIAS Measurements +
"""
Correlation-Heatmap: Influencing Variables Years 2018-2020
"""
#sample_bias = create_bias_data(sample_sp)
#corr = calculate_correlation_heatmap(sample_bias, title=title_name, source=source_both)

# + DESC Measurements
"""
Correlation-Heatmap: Fundamental Variables Years 2018-2020
"""
#sample_desc = create_descriptive_data(sample_sp)
#corr = calculate_correlation_heatmap(sample_bias, title=title_name, source=source_ref)


# +++++ RESULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++ OVERVIEW +++
"""
Chapter: 4.1.
"""
#overview_sent = import_sentiment_sheet('Overview', year=None)
#overview_sent_ref = import_sentiment_sheet('Overview_Refinitiv', year=None)
# ++ Descriptive Statistics ++
"""
Descriptive Statistics: Sentiment (ChatGPT and Refinitv) - Business Summary Prompt 
"""
#ds_overview_sent = calculate_descriptive_statistics(overview_sent, year=None)
#ds_overview_sent_ref = calculate_descriptive_statistics(overview_sent_ref, year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_overview_sent
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sample_desc_18, empty_df_one, ds_sample_desc_19, empty_df_two, ds_sample_desc_20], ignore_index=False)
#merged_df = pd.concat([ds_overview_sent, empty_df_one, ds_overview_sent_ref], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# ++ Correlations ++
"""
Correlation-Heatmap: Sentiment ChatGPT - Business Summary Prompt
"""
#corr = calculate_correlation_heatmap(overview_sent, title=title_name, source=source_ref, tight=False)
"""
Correlation-Heatmap: Sentiment Refinitiv - Business Summary Prompt
"""
#corr = calculate_correlation_heatmap(overview_sent_ref, title=title_name, source=source_ref, tight=False)
# + Names Count & Sector Distribution +
#overview_sent_bw = sent_perc_best_worst_sentiment(overview_sent)
#plot_sector_distribution_best_worst_overview = plot_sector_distribution_best_worst_lm_sent(sample_sp, overview_sent_bw, over_title="title")#
#overview_sent_bwcount = count_sector_and_names_best_worst_sent(sample_sp, overview_sent_bw, 'overview')

# +++ ESG +++
"""
Chapter: 4.2.
"""
# generally needed data imports
#sp_excel_data = import_sp_excel_data(excel_name='Quant_MA.xlsx')
#cleaned_labor_data = clean_sp_excel_data(sp_excel_data, clean_columns=True, labor_filter=False )
#cleaned_sp_data = clean_sp_excel_data(sp_excel_data, clean_columns=True)
#sample_sp = create_sp_sample_data(cleaned_sp_data)
#sample_bias = create_bias_data(sample_sp)
#sample_desc = create_descriptive_data(sample_sp)
# topic-specific question data imports
#sample_esg = create_esg_sample_data(sample_sp)
#sample_esg_2018 = sample_esg.loc[:, sample_esg.columns.str.contains('2018') | (sample_esg.columns == 'Common Name')]
#sample_esg_2019 = sample_esg.loc[:, sample_esg.columns.str.contains('2019') | (sample_esg.columns == 'Common Name')]
#sample_esg_2020 = sample_esg.loc[:, sample_esg.columns.str.contains('2020') | (sample_esg.columns == 'Common Name')]
# topic-specific question data imports (QUANTITATIVE)
#grade_esg_2018 = import_grades_sheet('ESG', year='2018')
#grade_esg_2019 = import_grades_sheet('ESG', year='2019')
#grade_esg_2020 = import_grades_sheet('ESG', year='2020')
# topic-specific question data imports (QUALITATIVE)
#sent_esg_2018 = import_sentiment_sheet('ESG', year='2018', corr=False)
#sent_esg_2019 = import_sentiment_sheet('ESG', year='2019', corr=False)
#sent_esg_2020 = import_sentiment_sheet('ESG', year='2020', corr=False)
# ++ QUANTITATIVE (ESG) ++
# + DESCRIPTIVE STATISTICS ESG (Grades & Quant-Variables) +
"""
Descriptive Statistics: Grades (and Deviations) and Refinitiv's Benchmark Variables - ESG Prompt 
"""
#quant_grades_esg_2018 = grade_sample_merge(sample_esg_2018, grade_esg_2018, ss=True, rq='ESG', year='2018')
#quant_grades_esg_2019 = grade_sample_merge(sample_esg_2019, grade_esg_2019, ss=True, rq='ESG', year='2019')
#quant_grades_esg_2020 = grade_sample_merge(sample_esg_2020, grade_esg_2020, ss=True, rq='ESG', year='2020')
#ds_quant_grades_esg_2018 = calculate_descriptive_statistics(quant_grades_esg_2018, year=None)
#ds_quant_grades_esg_2019 = calculate_descriptive_statistics(quant_grades_esg_2019, year=None)
#ds_quant_grades_esg_2020 = calculate_descriptive_statistics(quant_grades_esg_2020, year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_quant_grades_esg_2018
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_quant_grades_esg_2018, empty_df_one, ds_quant_grades_esg_2019, empty_df_two, ds_quant_grades_esg_2020], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# + CORRELATIONS ESG (Grades & Bias & Quant-Variables)
"""
Correlation-Heatmap: Grades (and Deviations), Refinitiv's Benchmark- and Influencing-Variables - ESG Prompt Year 2018-2020
"""
#corr_esg_2018_quant_grades = grade_sample_merge(sample_esg_2018, grade_esg_2018, ss=False, rq='ESG', year='2018')
#corr_esg_2019_quant_grades = grade_sample_merge(sample_esg_2019, grade_esg_2019, ss=False, rq='ESG', year='2019')
#corr_esg_2020_quant_grades = grade_sample_merge(sample_esg_2020, grade_esg_2020, ss=False, rq='ESG', year='2020')
#corr_esg_2018_quant_grades = calculate_correlation_heatmap(corr_esg_2018_quant_grades, title='', source='')
#corr_esg_2019_quant_grades = calculate_correlation_heatmap(corr_esg_2019_quant_grades, title='', source='')
#corr_esg_2020_quant_grades = calculate_correlation_heatmap(corr_esg_2020_quant_grades, title='', source='')
# + BEST & WORST QUANTITATIVE ESG (NAMES & SECTOR COUNT) +
#grade_esg_2018_bw = grades_perc_best_worst(quant_grades_esg_2018, sample_desc, sample_esg, rq='ESG', year='2018')
#grade_esg_2019_bw = grades_perc_best_worst(quant_grades_esg_2019, sample_desc, sample_esg, rq='ESG', year='2019')
#grade_esg_2020_bw = grades_perc_best_worst(quant_grades_esg_2020, sample_desc, sample_esg, rq='ESG', year='2020')
"""
Descriptive Statistics: Largest negative Deviations (positive Bias) - ESG Prompt Years 2018-2020
"""
#ds_grade_esg_2018_b = calculate_descriptive_statistics(grade_esg_2018_bw[1], year=None)
#ds_grade_esg_2019_b = calculate_descriptive_statistics(grade_esg_2019_bw[1], year=None)
#ds_grade_esg_2020_b = calculate_descriptive_statistics(grade_esg_2020_bw[1], year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_grade_esg_2018_b
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_grade_esg_2018_b, empty_df_one, ds_grade_esg_2019_b, empty_df_two, ds_grade_esg_2020_b], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
#merged_df = pd.concat([ds_grade_esg_2018_b, empty_df_one, ds_grade_esg_2019_b, empty_df_two, ds_grade_esg_2020_b], ignore_index=False)
"""
Descriptive Statistics: Largest positive Deviations (negative Bias) - ESG Prompt Years 2018-2020
"""
#ds_grade_esg_2018_w = calculate_descriptive_statistics(grade_esg_2018_bw[0], year=None)
#ds_grade_esg_2019_w = calculate_descriptive_statistics(grade_esg_2019_bw[0], year=None)
#ds_grade_esg_2020_w = calculate_descriptive_statistics(grade_esg_2020_bw[0], year=None)
#bench_df = ds_grade_esg_2018_w
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_grade_esg_2018_w, empty_df_one, ds_grade_esg_2019_w, empty_df_two, ds_grade_esg_2020_w], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# ++ SENTIMENT (ESG) ++
#sent_esg_2018 = import_sentiment_sheet('ESG', year='2018', corr=False)
#sent_esg_2019 = import_sentiment_sheet('ESG', year='2019', corr=False)
#sent_esg_2020 = import_sentiment_sheet('ESG', year='2020', corr=False)
# + DESCRIPTIVE STATISTICS ESG (QUALITATIVE) +
#ds_sent_esg_2018 = calculate_descriptive_statistics(sent_esg_2018, year='2018')
#ds_sent_esg_2019 = calculate_descriptive_statistics(sent_esg_2019, year='2019')
#ds_sent_esg_2020 = calculate_descriptive_statistics(sent_esg_2020, year='2020')
"""
Descriptive Statistics: Sentiment - ESG Prompt Years 2018-2020
"""
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sent_esg_2018
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_esg_2018, empty_df_one, ds_sent_esg_2019, empty_df_two, ds_sent_esg_2020], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
#sent_esg_2018 = import_sentiment_sheet('ESG', year='2018', corr=True)
#sent_esg_2019 = import_sentiment_sheet('ESG', year='2019', corr=True)
#sent_esg_2020 = import_sentiment_sheet('ESG', year='2020', corr=True)
# + CORRELATIONS ESG (QUALITATIVE & Bias-Variables)
"""
Correlation-Heatmap: Sentiment, Refinitiv's Benchmark- and Influecing-Variables - ESG Prompt Year 2018-2020
"""
#corr_sent_bias_esg_2020 = sample_esg_2018.merge(sent_esg_2018, left_on='Common Name', right_index=True)
#corr_sent_bias_esg_2020 = calculate_correlation_heatmap(corr_sent_bias_esg_2020, title='', source='')
#corr_sent_bias_esg_2019 = sample_esg_2019.merge(sent_esg_2019, left_on='Common Name', right_index=True)
#corr_sent_bias_esg_2019 = calculate_correlation_heatmap(corr_sent_bias_esg_2019, title='', source='')
#corr_sent_bias_esg_2020 = sample_esg_2020.merge(sent_esg_2020, left_on='Common Name', right_index=True)
#corr_sent_bias_esg_2020 = calculate_correlation_heatmap(corr_sent_bias_esg_2020, title='', source='')
# + CORRELATIONS ESG (Grades & Sentiment)
"""
Correlation-Heatmap: Sentiment and Grades (and Deviations) - ESG Prompt Years 2018-2020
"""
#sent_grade_esg_2018 = grade_sample_sent_merge(sample_esg_2018, grade_esg_2018, sent_esg_2018, rq='ESG', year='2018')
#sent_grade_esg_2019 = grade_sample_sent_merge(sample_esg_2019, grade_esg_2019, sent_esg_2019, rq='ESG', year='2019')
#sent_grade_esg_2020 = grade_sample_sent_merge(sample_esg_2020, grade_esg_2020, sent_esg_2020, rq='ESG', year='2020')
#sent_grades_esg = sent_grade_esg_2018.merge(sent_grade_esg_2019, left_index=True, right_index=True).merge(sent_grade_esg_2020, left_index=True, right_index=True)
#corr_sent_grades_esg = calculate_correlation_heatmap(sent_grades_esg, title='', source='')
# + BEST & WORST QUALITATIVE ESG (NAMES & SECTOR COUNT) +
#sent_esg_2018 = import_sentiment_sheet('ESG', year='2018', corr=False)
#sent_esg_2018_bw = sent_perc_best_worst_sentiment(sent_esg_2018)
#ent_esg_2019 = import_sentiment_sheet('ESG', year='2019', corr=False)
#sent_esg_2019_bw = sent_perc_best_worst_sentiment(sent_esg_2019)
#sent_esg_2020 = import_sentiment_sheet('ESG', year='2020', corr=False)
#sent_esg_2020_bw = sent_perc_best_worst_sentiment(sent_esg_2020)
#sent_esg_2018_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_esg_2018_bw, 'ESG_18')
#sent_esg_2019_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_esg_2019_bw, 'ESG_19')
#sent_esg_2020_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_esg_2020_bw, 'ESG_20')
# + ADDITIONAL MEASUREMENTS ESG (Best & Worst QUALITATIVE) +
"""
Descriptive Statistics: Most positive Sentiment - ESG Prompt Years 2018-2020
"""
#sent_esg_2018_bw = sent_perc_best_worst_sentiment(sent_esg_2018)
#sent_esg_2018_b = merge_common_index(sample_desc, sent_esg_2018_bw[0])
#ds_sent_esg_2018_b = calculate_descriptive_statistics(sent_esg_2018_b, year='2018')
#sent_esg_2019_bw = sent_perc_best_worst_sentiment(sent_esg_2019)
#sent_esg_2019_b = merge_common_index(sample_desc, sent_esg_2019_bw[0])
#ds_sent_esg_2019_b = calculate_descriptive_statistics(sent_esg_2019_b, year='2019')
#sent_esg_2020_bw = sent_perc_best_worst_sentiment(sent_esg_2020)
#sent_esg_2020_b = merge_common_index(sample_desc, sent_esg_2020_bw[0])
#ds_sent_esg_2020_b = calculate_descriptive_statistics(sent_esg_2020_b, year='2020')
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sent_esg_2018_b
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_esg_2018_b, empty_df_one, ds_sent_esg_2019_b, empty_df_two, ds_sent_esg_2020_b], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
"""
Descriptive Statistics: Most negative Sentiment - ESG Prompt Years 2018-2020
"""
#sent_esg_2018_w = merge_common_index(sample_desc, sent_esg_2018_bw[1])
#ds_sent_esg_2018_w = calculate_descriptive_statistics(sent_esg_2018_w, year='2018')
#sent_esg_2019_w = merge_common_index(sample_desc, sent_esg_2019_bw[1])
#ds_sent_esg_2019_w = calculate_descriptive_statistics(sent_esg_2019_w, year='2019')
#sent_esg_2020_w = merge_common_index(sample_desc, sent_esg_2020_bw[1])
#ds_sent_esg_2020_w = calculate_descriptive_statistics(sent_esg_2020_w, year='2020')
#bench_df = ds_sent_esg_2018_w
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_esg_2018_w, empty_df_one, ds_sent_esg_2019_w, empty_df_two, ds_sent_esg_2020_w], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#   f.write(table_html_2)

# +++ ROBUSTNESS-TESTS (ESG) +++
# import all robustness-test sheet's (and manually check is get_grade() really caught all grades)
#ESG_19_I1 = get_grade(sheet_name='ESG_RT1_2019')
#ESG_19_I2 = get_grade(sheet_name='ESG_RT2_2019')
#ESG_19_I2 = get_grade(sheet_name='ESG_RT3_2019')
#ESG_19_I3 = get_grade(sheet_name='ESG_RT4_2019')
#ESG_RT1 = import_grades_sheet('ESG_RT1', year='2019')
#ESG_RT2 = import_grades_sheet('ESG_RT2', year='2019')
#ESG_RT3 = import_grades_sheet('ESG_RT3', year='2019')
#ESG_RT4 = import_grades_sheet('ESG_RT4', year='2019')

#grade_esg_2019 = import_grades_sheet('ESG', year='2019')
#print(grade_esg_2019)
#df = pd.concat([ESG_RT1, ESG_RT2, ESG_RT3, ESG_RT4, grade_esg_2019], axis=1)
#with pd.ExcelWriter("Grades_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#        df.to_excel(writer, sheet_name='ESG_Rob', startrow=0, startcol=0)


# +++ Innovation +++
"""
Chapter: 4.3.
"""
# generally needed data imports
#sp_excel_data = import_sp_excel_data(excel_name='Quant_MA.xlsx')
#cleaned_labor_data = clean_sp_excel_data(sp_excel_data, clean_columns=True, labor_filter=False )
#cleaned_sp_data = clean_sp_excel_data(sp_excel_data, clean_columns=True)
#sample_sp = create_sp_sample_data(cleaned_sp_data)
#sample_bias = create_bias_data(sample_sp)
#sample_desc = create_descriptive_data(sample_sp)
# topic-specific question data imports
#sample_innovation = create_inno_sample_data(sample_sp)
#sample_innovation = sample_innovation[0]
#sample_innovation_2018 = sample_innovation.loc[:, sample_innovation.columns.str.contains('2018') | (sample_innovation.columns == 'Common Name')]
#sample_innovation_2019 = sample_innovation.loc[:, sample_innovation.columns.str.contains('2019') | (sample_innovation.columns == 'Common Name')]
#sample_innovation_2020 = sample_innovation.loc[:, sample_innovation.columns.str.contains('2020') | (sample_innovation.columns == 'Common Name')]
# topic-specific question data imports (QUANTITATIVE)
#grade_innovation_2018 = import_grades_sheet('Innovation', year='2018')
#grade_innovation_2019 = import_grades_sheet('Innovation', year='2019')
#grade_innovation_2020 = import_grades_sheet('Innovation', year='2020')
# topic-specific question data imports (QUALITATIVE)
#sent_innovation_2018 = import_sentiment_sheet('Innovation', year='2018', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2019 = import_sentiment_sheet('Innovation', year='2019', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2020 = import_sentiment_sheet('Innovation', year='2020', corr=False, adapt=True, adapt_df=sample_innovation)
# ++ QUANTITATIVE (Innovation) ++
# + DESCRIPTIVE STATISTICS Innovation (Grades & Quant-Variables) +
"""
Descriptive Statistics: Grades (and Deviations) and Refinitiv's Benchmark Variables - Innovation Prompt 
"""
#quant_grades_innovation_2018 = grade_sample_merge(sample_innovation_2018, grade_innovation_2018, ss=True, rq='Innovation', year='2018')
#quant_grades_innovation_2019 = grade_sample_merge(sample_innovation_2019, grade_innovation_2019, ss=True, rq='Innovation', year='2019')
#quant_grades_innovation_2020 = grade_sample_merge(sample_innovation_2020, grade_innovation_2020, ss=True, rq='Innovation', year='2020')
#ds_quant_grades_innovation_2018 = calculate_descriptive_statistics(quant_grades_innovation_2018, year=None)
#ds_quant_grades_innovation_2019 = calculate_descriptive_statistics(quant_grades_innovation_2019, year=None)
#ds_quant_grades_innovation_2020 = calculate_descriptive_statistics(quant_grades_innovation_2020, year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_quant_grades_innovation_2018
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_quant_grades_innovation_2018, empty_df_one, ds_quant_grades_innovation_2019, empty_df_two, ds_quant_grades_innovation_2020], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# + CORRELATIONS Innovation (Grades & Bias & Quant-Variables)
"""
Correlation-Heatmap: Grades (and Deviations), Refinitiv's Benchmark- and Influencing-Variables - Innovation Prompt Year 2018-2020
"""
#corr_innovation_2018_quant_grades = grade_sample_merge(sample_innovation_2018, grade_innovation_2018, ss=False, rq='Innovation', year='2018')
#corr_innovation_2019_quant_grades = grade_sample_merge(sample_innovation_2019, grade_innovation_2019, ss=False, rq='Innovation', year='2019')
#corr_innovation_2020_quant_grades = grade_sample_merge(sample_innovation_2020, grade_innovation_2020, ss=False, rq='Innovation', year='2020')
#corr_innovation_2018_quant_grades = calculate_correlation_heatmap(corr_innovation_2018_quant_grades, title='', source='', tight=True)
#corr_innovation_2019_quant_grades = calculate_correlation_heatmap(corr_innovation_2019_quant_grades, title='', source='', tight=True)
#corr_innovation_2020_quant_grades = calculate_correlation_heatmap(corr_innovation_2020_quant_grades, title='', source='', tight=True)
# + BEST & WORST QUANTITATIVE Innovation (NAMES & SECTOR COUNT) +
#grade_innovation_2018_bw = grades_perc_best_worst(quant_grades_innovation_2018, sample_desc, sample_innovation, rq='Innovation', year='2018')
#grade_innovation_2019_bw = grades_perc_best_worst(quant_grades_innovation_2019, sample_desc, sample_innovation, rq='Innovation', year='2019')
#grade_innovation_2020_bw = grades_perc_best_worst(quant_grades_innovation_2020, sample_desc, sample_innovation, rq='Innovation', year='2020')
"""
Descriptive Statistics: Largest negative Deviations (positive Bias) - Innovation Prompt Years 2018-2020
"""
#ds_grade_innovation_2018_b = calculate_descriptive_statistics(grade_innovation_2018_bw[1], year=None)
#ds_grade_innovation_2019_b = calculate_descriptive_statistics(grade_innovation_2019_bw[1], year=None)
#ds_grade_innovation_2020_b = calculate_descriptive_statistics(grade_innovation_2020_bw[1], year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_grade_innovation_2018_b
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_grade_innovation_2018_b, empty_df_one, ds_grade_innovation_2019_b, empty_df_two, ds_grade_innovation_2020_b], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#   f.write(table_html_2)
#merged_df = pd.concat([ds_grade_innovation_2018_b, empty_df_one, ds_grade_innovation_2019_b, empty_df_two, ds_grade_innovation_2020_b], ignore_index=False)
"""
Descriptive Statistics: Largest positive Deviations (negative Bias) - Innovation Prompt Years 2018-2020
"""
#ds_grade_innovation_2018_w = calculate_descriptive_statistics(grade_innovation_2018_bw[0], year=None)
#ds_grade_innovation_2019_w = calculate_descriptive_statistics(grade_innovation_2019_bw[0], year=None)
#ds_grade_innovation_2020_w = calculate_descriptive_statistics(grade_innovation_2020_bw[0], year=None)
#bench_df = ds_grade_innovation_2018_w
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_grade_innovation_2018_w, empty_df_one, ds_grade_innovation_2019_w, empty_df_two, ds_grade_innovation_2020_w], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# ++ SENTIMENT (Innovation) ++
#sent_innovation_2018 = import_sentiment_sheet('Innovation', year='2018', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2019 = import_sentiment_sheet('Innovation', year='2019', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2020 = import_sentiment_sheet('Innovation', year='2020', corr=False, adapt=True, adapt_df=sample_innovation)
# + DESCRIPTIVE STATISTICS Innovation (QUALITATIVE) +
#ds_sent_innovation_2018 = calculate_descriptive_statistics(sent_innovation_2018, year='2018')
#ds_sent_innovation_2019 = calculate_descriptive_statistics(sent_innovation_2019, year='2019')
#ds_sent_innovation_2020 = calculate_descriptive_statistics(sent_innovation_2020, year='2020')
"""
Descriptive Statistics: Sentiment - Innovation Prompt Years 2018-2020
"""
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sent_innovation_2018
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_innovation_2018, empty_df_one, ds_sent_innovation_2019, empty_df_two, ds_sent_innovation_2020], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#   f.write(table_html_2)
#sent_innovation_2018 = import_sentiment_sheet('Innovation', year='2018', corr=True, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2019 = import_sentiment_sheet('Innovation', year='2019', corr=True, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2020 = import_sentiment_sheet('Innovation', year='2020', corr=True, adapt=True, adapt_df=sample_innovation)
# + CORRELATIONS Innovation (QUALITATIVE & Bias-Variables)
"""
Correlation-Heatmap: Sentiment, Refinitiv's Benchmark- and Influecing-Variables - Innovation Prompt Year 2018-2020
"""
#corr_sent_bias_innovation_2020 = sample_innovation_2018.merge(sent_innovation_2018, left_on='Common Name', right_index=True)
#corr_sent_bias_innovation_2020 = calculate_correlation_heatmap(corr_sent_bias_innovation_2020, title='', source='')
#corr_sent_bias_innovation_2019 = sample_innovation_2019.merge(sent_innovation_2019, left_on='Common Name', right_index=True)
#corr_sent_bias_innovation_2019 = calculate_correlation_heatmap(corr_sent_bias_innovation_2019, title='', source='')
#corr_sent_bias_innovation_2020 = sample_innovation_2020.merge(sent_innovation_2020, left_on='Common Name', right_index=True)
#corr_sent_bias_innovation_2020 = calculate_correlation_heatmap(corr_sent_bias_innovation_2020, title='', source='')
# + CORRELATIONS Innovation (Grades & Sentiment)
"""
Correlation-Heatmap: Sentiment and Grades (and Deviations) - Innovation Prompt Years 2018-2020
"""
#sent_grade_innovation_2018 = grade_sample_sent_merge(sample_innovation_2018, grade_innovation_2018, sent_innovation_2018, rq='Innovation', year='2018')
#sent_grade_innovation_2019 = grade_sample_sent_merge(sample_innovation_2019, grade_innovation_2019, sent_innovation_2019, rq='Innovation', year='2019')
#sent_grade_innovation_2020 = grade_sample_sent_merge(sample_innovation_2020, grade_innovation_2020, sent_innovation_2020, rq='Innovation', year='2020')
#sent_grades_innovation = sent_grade_innovation_2018.merge(sent_grade_innovation_2019, left_index=True, right_index=True).merge(sent_grade_innovation_2020, left_index=True, right_index=True)
#corr_sent_grades_innovation = calculate_correlation_heatmap(sent_grades_innovation, title='', source='', tight=True)

# + BEST & WORST QUALITATIVE Innovation (NAMES & SECTOR COUNT) +
#sent_innovation_2018 = import_sentiment_sheet('Innovation', year='2018', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2018_bw = sent_perc_best_worst_sentiment(sent_innovation_2018)
#sent_innovation_2019 = import_sentiment_sheet('Innovation', year='2019', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2019_bw = sent_perc_best_worst_sentiment(sent_innovation_2019)
#sent_innovation_2020 = import_sentiment_sheet('Innovation', year='2020', corr=False, adapt=True, adapt_df=sample_innovation)
#sent_innovation_2020_bw = sent_perc_best_worst_sentiment(sent_innovation_2020)
#sent_innovation_2018_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_innovation_2018_bw, 'Innovation_18')
#sent_innovation_2019_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_innovation_2019_bw, 'Innovation_19')
#sent_innovation_2020_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_innovation_2020_bw, 'Innovation_20')
# + ADDITIONAL MEASUREMENTS Innovation (Best & Worst QUALITATIVE) +
"""
Descriptive Statistics: Most positive Sentiment - Innovation Prompt Years 2018-2020
"""
#sent_innovation_2018_bw = sent_perc_best_worst_sentiment(sent_innovation_2018)
#sent_innovation_2018_b = merge_common_index(sample_desc, sent_innovation_2018_bw[0])
#ds_sent_innovation_2018_b = calculate_descriptive_statistics(sent_innovation_2018_b, year='2018')
#sent_innovation_2019_bw = sent_perc_best_worst_sentiment(sent_innovation_2019)
#sent_innovation_2019_b = merge_common_index(sample_desc, sent_innovation_2019_bw[0])
#ds_sent_innovation_2019_b = calculate_descriptive_statistics(sent_innovation_2019_b, year='2019')
#sent_innovation_2020_bw = sent_perc_best_worst_sentiment(sent_innovation_2020)
#sent_innovation_2020_b = merge_common_index(sample_desc, sent_innovation_2020_bw[0])
#ds_sent_innovation_2020_b = calculate_descriptive_statistics(sent_innovation_2020_b, year='2020')
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sent_innovation_2018_b
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_innovation_2018_b, empty_df_one, ds_sent_innovation_2019_b, empty_df_two, ds_sent_innovation_2020_b], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
"""
Descriptive Statistics: Most negative Sentiment - Innovation Prompt Years 2018-2020
"""
#sent_innovation_2018_bw = sent_perc_best_worst_sentiment(sent_innovation_2018)
#sent_innovation_2018_w = merge_common_index(sample_desc, sent_innovation_2018_bw[1])
#ds_sent_innovation_2018_w = calculate_descriptive_statistics(sent_innovation_2018_w, year='2018')
#sent_innovation_2019_bw = sent_perc_best_worst_sentiment(sent_innovation_2019)
#sent_innovation_2019_w = merge_common_index(sample_desc, sent_innovation_2019_bw[1])
#ds_sent_innovation_2019_w = calculate_descriptive_statistics(sent_innovation_2019_w, year='2019')
#sent_innovation_2020_bw = sent_perc_best_worst_sentiment(sent_innovation_2020)
#sent_innovation_2020_w = merge_common_index(sample_desc, sent_innovation_2020_bw[1])
#ds_sent_innovation_2020_w = calculate_descriptive_statistics(sent_innovation_2020_w, year='2020')
#bench_df = ds_sent_innovation_2018_w
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_innovation_2018_w, empty_df_one, ds_sent_innovation_2019_w, empty_df_two, ds_sent_innovation_2020_w], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#   f.write(table_html_2)


# +++ Labor +++
"""
Chapter: 4.4.
"""
# generally needed data imports
sp_excel_data = import_sp_excel_data(excel_name='Quant_MA.xlsx')
cleaned_labor_data = clean_sp_excel_data(sp_excel_data, clean_columns=True, labor_filter=False )
cleaned_sp_data = clean_sp_excel_data(sp_excel_data, clean_columns=True)
sample_sp = create_sp_sample_data(cleaned_sp_data)
sample_bias = create_bias_data(sample_sp)
sample_desc = create_descriptive_data(sample_sp)
# topic-specific question data imports
sample_labor = create_labor_sample_data(cleaned_labor_data, sample_sp)
sample_labor_2018 = sample_labor.loc[:, sample_labor.columns.str.contains('2018') | (sample_labor.columns == 'Common Name')]
sample_labor_2019 = sample_labor.loc[:, sample_labor.columns.str.contains('2019') | (sample_labor.columns == 'Common Name')]
sample_labor_2020 = sample_labor.loc[:, sample_labor.columns.str.contains('2020') | (sample_labor.columns == 'Common Name')]
# topic-specific question data imports (QUANTITATIVE)
grade_labor_2018 = import_grades_sheet('Labor', year='2018')
grade_labor_2019 = import_grades_sheet('Labor', year='2019')
grade_labor_2020 = import_grades_sheet('Labor', year='2020')
# topic-specific question data imports (QUALITATIVE)
sent_labor_2018 = import_sentiment_sheet('Labor', year='2018', corr=False, adapt=True, adapt_df=sample_labor)
sent_labor_2019 = import_sentiment_sheet('Labor', year='2019', corr=False, adapt=True, adapt_df=sample_labor)
sent_labor_2020 = import_sentiment_sheet('Labor', year='2020', corr=False, adapt=True, adapt_df=sample_labor)
# ++ QUANTITATIVE (Labor) ++
# + DESCRIPTIVE STATISTICS Labor (Grades & Quant-Variables) +
"""
Descriptive Statistics: Grades (and Deviations) and Refinitiv's Benchmark Variables - Labor Prompt 
"""
#quant_grades_labor_2018 = grade_sample_merge(sample_labor_2018, grade_labor_2018, ss=True, rq='Labor', year='2018')
#quant_grades_labor_2019 = grade_sample_merge(sample_labor_2019, grade_labor_2019, ss=True, rq='Labor', year='2019')
#quant_grades_labor_2020 = grade_sample_merge(sample_labor_2020, grade_labor_2020, ss=True, rq='Labor', year='2020')
#ds_quant_grades_labor_2018 = calculate_descriptive_statistics(quant_grades_labor_2018, year=None)
#ds_quant_grades_labor_2019 = calculate_descriptive_statistics(quant_grades_labor_2019, year=None)
#ds_quant_grades_labor_2020 = calculate_descriptive_statistics(quant_grades_labor_2020, year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_quant_grades_labor_2018
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_quant_grades_labor_2018, empty_df_one, ds_quant_grades_labor_2019, empty_df_two, ds_quant_grades_labor_2020], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# + CORRELATIONS Labor (Grades & Bias & Quant-Variables)
"""
Correlation-Heatmap: Grades (and Deviations), Refinitiv's Benchmark- and Influencing-Variables - Labor Prompt Year 2018-2020
"""
#corr_labor_2018_quant_grades = grade_sample_merge(sample_labor_2018, grade_labor_2018, ss=False, rq='Labor', year='2018')
#corr_labor_2019_quant_grades = grade_sample_merge(sample_labor_2019, grade_labor_2019, ss=False, rq='Labor', year='2019')
#corr_labor_2020_quant_grades = grade_sample_merge(sample_labor_2020, grade_labor_2020, ss=False, rq='Labor', year='2020')
#corr_labor_2018_quant_grades = calculate_correlation_heatmap(corr_labor_2018_quant_grades, title='', source='', tight=True)
#corr_labor_2019_quant_grades = calculate_correlation_heatmap(corr_labor_2019_quant_grades, title='', source='', tight=True)
#corr_labor_2020_quant_grades = calculate_correlation_heatmap(corr_labor_2020_quant_grades, title='', source='', tight=True)
# + BEST & WORST QUANTITATIVE Labor (NAMES & SECTOR COUNT) +
#grade_labor_2018_bw = grades_perc_best_worst(quant_grades_labor_2018, sample_desc, sample_labor, rq='Labor', year='2018')
#grade_labor_2019_bw = grades_perc_best_worst(quant_grades_labor_2019, sample_desc, sample_labor, rq='Labor', year='2019')
#grade_labor_2020_bw = grades_perc_best_worst(quant_grades_labor_2020, sample_desc, sample_labor, rq='Labor', year='2020')
"""
Descriptive Statistics: Largest negative Deviations (positive Bias) - Labor Prompt Years 2018-2020
"""
#ds_grade_labor_2018_b = calculate_descriptive_statistics(grade_labor_2018_bw[1], year=None)
#ds_grade_labor_2019_b = calculate_descriptive_statistics(grade_labor_2019_bw[1], year=None)
#ds_grade_labor_2020_b = calculate_descriptive_statistics(grade_labor_2020_bw[1], year=None)
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_grade_labor_2018_b
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_grade_labor_2018_b, empty_df_one, ds_grade_labor_2019_b, empty_df_two, ds_grade_labor_2020_b], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#   f.write(table_html_2)
"""
Descriptive Statistics: Largest positive Deviations (negative Bias) - Labor Prompt Years 2018-2020
"""
#ds_grade_labor_2018_w = calculate_descriptive_statistics(grade_labor_2018_bw[0], year=None)
#ds_grade_labor_2019_w = calculate_descriptive_statistics(grade_labor_2019_bw[0], year=None)
#ds_grade_labor_2020_w = calculate_descriptive_statistics(grade_labor_2020_bw[0], year=None)
#bench_df = ds_grade_labor_2018_w
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_grade_labor_2018_w, empty_df_one, ds_grade_labor_2019_w, empty_df_two, ds_grade_labor_2020_w], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
# ++ SENTIMENT (Labor) ++
sent_labor_2018 = import_sentiment_sheet('Labor', year='2018', corr=False, adapt=True, adapt_df=sample_labor)
sent_labor_2019 = import_sentiment_sheet('Labor', year='2019', corr=False, adapt=True, adapt_df=sample_labor)
sent_labor_2020 = import_sentiment_sheet('Labor', year='2020', corr=False, adapt=True, adapt_df=sample_labor)
# + DESCRIPTIVE STATISTICS Labor (QUALITATIVE) +
#ds_sent_labor_2018 = calculate_descriptive_statistics(sent_labor_2018, year='2018')
#ds_sent_labor_2019 = calculate_descriptive_statistics(sent_labor_2019, year='2019')
#ds_sent_labor_2020 = calculate_descriptive_statistics(sent_labor_2020, year='2020')
"""
Descriptive Statistics: Sentiment - Labor Prompt Years 2018-2020
"""
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sent_labor_2018
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_labor_2018, empty_df_one, ds_sent_labor_2019, empty_df_two, ds_sent_labor_2020], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
#sent_labor_2018 = import_sentiment_sheet('Labor', year='2018', corr=True, adapt=True, adapt_df=sample_labor)
#sent_labor_2019 = import_sentiment_sheet('Labor', year='2019', corr=True, adapt=True, adapt_df=sample_labor)
#sent_labor_2020 = import_sentiment_sheet('Labor', year='2020', corr=True, adapt=True, adapt_df=sample_labor)
# + CORRELATIONS Labor (QUALITATIVE & Bias-Variables)
"""
Correlation-Heatmap: Sentiment, Refinitiv's Benchmark- and Influecing-Variables - Labor Prompt Year 2018-2020
"""
#corr_sent_bias_labor_2020 = sample_labor_2018.merge(sent_labor_2018, left_on='Common Name', right_index=True)
#corr_sent_bias_labor_2020 = calculate_correlation_heatmap(corr_sent_bias_labor_2020, title='', source='')
#corr_sent_bias_labor_2019 = sample_labor_2019.merge(sent_labor_2019, left_on='Common Name', right_index=True)
#corr_sent_bias_labor_2019 = calculate_correlation_heatmap(corr_sent_bias_labor_2019, title='', source='')
#corr_sent_bias_labor_2020 = sample_labor_2020.merge(sent_labor_2020, left_on='Common Name', right_index=True)
#corr_sent_bias_labor_2020 = calculate_correlation_heatmap(corr_sent_bias_labor_2020, title='', source='')
# + CORRELATIONS Labor (Grades & Sentiment)
"""
Correlation-Heatmap: Sentiment and Grades (and Deviations) - Labor Prompt Years 2018-2020
"""
#sent_grade_labor_2018 = grade_sample_sent_merge(sample_labor_2018, grade_labor_2018, sent_labor_2018, rq='Labor', year='2018')
#sent_grade_labor_2019 = grade_sample_sent_merge(sample_labor_2019, grade_labor_2019, sent_labor_2019, rq='Labor', year='2019')
#sent_grade_labor_2020 = grade_sample_sent_merge(sample_labor_2020, grade_labor_2020, sent_labor_2020, rq='Labor', year='2020')
#sent_grades_labor = sent_grade_labor_2018.merge(sent_grade_labor_2019, left_index=True, right_index=True).merge(sent_grade_labor_2020, left_index=True, right_index=True)
#corr_sent_grades_labor = calculate_correlation_heatmap(sent_grades_labor, title='', source='', tight=True)
# + BEST & WORST QUALITATIVE Labor (NAMES & SECTOR COUNT) +
#sent_labor_2018 = import_sentiment_sheet('Labor', year='2018', corr=False, adapt=True, adapt_df=sample_labor)
#sent_labor_2018_bw = sent_perc_best_worst_sentiment(sent_labor_2018)
#sent_labor_2019 = import_sentiment_sheet('Labor', year='2019', corr=False, adapt=True, adapt_df=sample_labor)
#sent_labor_2019_bw = sent_perc_best_worst_sentiment(sent_labor_2019)
#sent_labor_2020 = import_sentiment_sheet('Labor', year='2020', corr=False, adapt=True, adapt_df=sample_labor)
#sent_labor_2020_bw = sent_perc_best_worst_sentiment(sent_labor_2020)
#sent_labor_2018_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_labor_2018_bw, 'Labor_18')
#sent_labor_2019_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_labor_2019_bw, 'Labor_19')
#sent_labor_2020_bwcount = count_sector_and_names_best_worst_sent(sample_sp, sent_labor_2020_bw, 'Labor_20')
# + ADDITIONAL MEASUREMENTS Labor (Best & Worst QUALITATIVE) +
"""
Descriptive Statistics: Most positive Sentiment - Labor Prompt Years 2018-2020
"""
#sent_labor_2018_bw = sent_perc_best_worst_sentiment(sent_labor_2018)
#sent_labor_2018_b = merge_common_index(sample_desc, sent_labor_2018_bw[0])
#ds_sent_labor_2018_b = calculate_descriptive_statistics(sent_labor_2018_b, year='2018')
#sent_labor_2019_bw = sent_perc_best_worst_sentiment(sent_labor_2019)
#sent_labor_2019_b = merge_common_index(sample_desc, sent_labor_2019_bw[0])
#ds_sent_labor_2019_b = calculate_descriptive_statistics(sent_labor_2019_b, year='2019')
#sent_labor_2020_bw = sent_perc_best_worst_sentiment(sent_labor_2020)
#sent_labor_2020_b = merge_common_index(sample_desc, sent_labor_2020_bw[0])
#ds_sent_labor_2020_b = calculate_descriptive_statistics(sent_labor_2020_b, year='2020')
# create empty df's for ep (this is used like a copy-pased-tool in order to produce the actual tables)
#bench_df = ds_sent_labor_2018_b
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_labor_2018_b, empty_df_one, ds_sent_labor_2019_b, empty_df_two, ds_sent_labor_2020_b], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
"""
Descriptive Statistics: Most negative Sentiment - Labor Prompt Years 2018-2020
"""
#sent_labor_2018_bw = sent_perc_best_worst_sentiment(sent_labor_2018)
#sent_labor_2018_w = merge_common_index(sample_desc, sent_labor_2018_bw[1])
#ds_sent_labor_2018_w = calculate_descriptive_statistics(sent_labor_2018_w, year='2018')
#sent_labor_2019_bw = sent_perc_best_worst_sentiment(sent_labor_2019)
#sent_labor_2019_w = merge_common_index(sample_desc, sent_labor_2019_bw[1])
#ds_sent_labor_2019_w = calculate_descriptive_statistics(sent_labor_2019_w, year='2019')
#sent_labor_2020_bw = sent_perc_best_worst_sentiment(sent_labor_2020)
#sent_labor_2020_w = merge_common_index(sample_desc, sent_labor_2020_bw[1])
#ds_sent_labor_2020_w = calculate_descriptive_statistics(sent_labor_2020_w, year='2020')
#bench_df = ds_sent_labor_2018_w
#empty_df_one = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_one.index = [' ']
#empty_df_two = pd.DataFrame({col: [''] * len(bench_df.columns) for col in bench_df.columns}).iloc[0:1]
#empty_df_two.index = ['']
#merged_df = pd.concat([ds_sent_labor_2018_w, empty_df_one, ds_sent_labor_2019_w, empty_df_two, ds_sent_labor_2020_w], ignore_index=False)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=merged_df, title='', source='')
#with open('ATable.html', 'w') as f:
#   f.write(table_html_2)



# +++ Return +++
"""
Chapter: 4.5.
"""
# generally needed data imports
#sp_excel_data = import_sp_excel_data(excel_name='Quant_MA.xlsx')
#cleaned_sp_data = clean_sp_excel_data(sp_excel_data, clean_columns=True)
#sample_sp = create_sp_sample_data(cleaned_sp_data)
#sample_bias = create_bias_data(sample_sp)
#sample_desc = create_descriptive_data(sample_sp)
# topic-specific question data imports
#sample_return = create_performance_sample_data(sample_sp)
#print(sample_return)
#grade_return = import_reply_sheet("Responses_MA.xlsx", "Performance")
#grade_return["Common Name"] = grade_return["company_name"]
#grade_return = grade_return.set_index("Common Name")
#print(grade_return)
#sent_return = import_sentiment_sheet('Performance', year=None, corr=False)
#print(sent_return)
# ++ QUANTITATIVE and QUALITATIVE (Performance) ++
# + DESCRIPTIVE STATISTICS Return(Grades & Quant-Variables) +
"""
Descriptive Statistics: Grades (and Deviations) and Refinitiv's Benchmark Variables - Return Prompt 
"""
#quant_grades_return = grade_sample_merge(sample_return, grade_return, ss=False, rq='Performance', year='')
#with pd.ExcelWriter("Further Findings.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#    quant_grades_return.to_excel(writer, sheet_name='Performance_Quant', startrow=0, startcol=0)
#ds_quant_grades_return = calculate_descriptive_statistics(quant_grades_return, year=None)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=ds_quant_grades_return, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)
"""
Descriptive Statistics: Sentiment - Return Prompt 
"""
#ds_sent_return = calculate_descriptive_statistics(sent_return, year='')
#with pd.ExcelWriter("Further Findings.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#    sent_return.to_excel(writer, sheet_name='Performance_Qual', startrow=0, startcol=0)
# plot the merged-df and create .hmtl file
#table_html_2 = plot_table(df=ds_sent_return, title='', source='')
#with open('ATable.html', 'w') as f:
#    f.write(table_html_2)


