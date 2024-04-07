from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import json
import pandas as pd
import re
from datetime import date
import time
#from ma import import_ma_excel_data, min_max_scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

"""
As the data from Google is freely available but often and unfortunately quickly reaches query limits, manual smaller iteration steps had to be chosen (e.g., 0, 100, and subsequently 101, 200, etc.).
The functions ending with wobnech (without benchmarking) represent test-functions. Only the benchmarking ones have been used in the papers (as described).
"""

#df_ma_data = import_ma_excel_data(excel_name='S&P500_Data_MA.xlsx')

#pt = TrendReq(hl="en-US", tz=360)
#pt = TrendReq()

def gsv_ticker(df_ma):
    df = pd.DataFrame()
    df_reset = df_ma.reset_index(drop=True)
    df_reset = df_reset[['Ticker','Common_Name']]

    for i in range(0, 100):

        successful = False
        while not successful:
            try:
                if df_reset['Ticker'][i] == 'MSFT':
                    pt.build_payload('MSFT', geo='US', timeframe='2020-01-01 2020-12-31')
                else:
                    pt.build_payload(['MSFT', df_reset['Ticker'][i]], geo='US',
                                     timeframe='2020-01-01 2020-12-31')

                iot = pt.interest_over_time()
                print(i)

                df[df_reset['Ticker'][i]] = iot.iloc[:, 1]
                successful = True

            except TooManyRequestsError:
                print("Too many requests. Waiting and then retrying...")
                time.sleep(30)
                continue

    with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=('gsv_ticker'), startrow=1, startcol=1)

    return df

#print(gsv_ticker(df_ma=df_ma_data))

def gsv_names_wbench(df_ma):
    df = pd.DataFrame()
    df_reset = df_ma.reset_index(drop=True)
    df_reset = df_reset[['Ticker', 'Common_Name']]

    for i in range(400, 491):

        successful = False
        while not successful:
            try:
                if df_reset['Common_Name'][i] == 'Microsoft Corp':
                    pt.build_payload(['Microsoft Corp'], geo='US', timeframe='2018-01-01 2018-12-31')
                else:
                    pt.build_payload(['Microsoft Corp', df_reset['Common_Name'][i]], geo='US',
                                     timeframe='2018-01-01 2018-12-31')

                iot = pt.interest_over_time()
                print(i)
    #            print(iot)
    #            print(iot.iloc[:, 1])

                if df_reset['Common_Name'][i] == 'Microsoft Corp':
                    df[df_reset['Common_Name'][i]] = iot.iloc[:, 0]

                else:
                    df[df_reset['Common_Name'][i]] = iot.iloc[:, 1]

                successful = True

            except TooManyRequestsError:
                print("Too many requests. Waiting and then retrying...")
                time.sleep(30)
                continue

    with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=('gsv_names'), startrow=1, startcol=1)

    return df

#print(gsv_names_wbench(df_ma=df_ma_data))

def gsv_names_wobench(df_ma):
    df = pd.DataFrame()
    df_reset = df_ma.reset_index(drop=True)
    df_reset = df_reset[['Ticker', 'Common_Name']]

    for i in range(0, 100):

        successful = False
        while not successful:
            try:
                pt.build_payload([df_reset['Common_Name'][i]], geo='US',
                                     timeframe='2019-01-01 2019-12-31')

                iot = pt.interest_over_time()
                print(i)

                if not iot.empty:
                    df[df_reset['Common_Name'][i]] = iot.iloc[:, 0]
                else:
                    df[df_reset['Common_Name'][i]] = [0] * len(df)

                successful = True

            except TooManyRequestsError:
                print("Too many requests. Waiting and then retrying...")
                time.sleep(30)
                continue

    with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=('gsv_names'), startrow=1, startcol=1)

    return df

#print(gsv_names_wobench(df_ma=df_ma_data))

def gsv_chatgpt_wobench():
    df = pd.DataFrame()

    pt.build_payload(['ChatGPT'], geo='US', timeframe='2020-01-01 2020-12-31')
    iot = pt.interest_over_time()
    print(iot)

    df['ChatGPT'] = iot.iloc[:, 0]

    with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=('gsv_ticker'), startrow=1, startcol=1)

    return df

#print(gsv_chatgpt_wbench())

def gsv_chatgpt_wbench():
    df = pd.DataFrame()

    pt.build_payload(['Microsoft Corp','ChatGPT'], geo='US', timeframe='2018-01-01 2018-12-31')
    iot = pt.interest_over_time()
    print(iot)

    df['ChatGPT'] = iot.iloc[:, 0]

    with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=('gsv_ticker'), startrow=1, startcol=1)

    return df

#print(gsv_chatgpt_wbench())

def import_gsv_sheet(sheet, date=True):
    gsv_df = pd.read_excel('GSV_MA.xlsx', sheet_name=sheet)
    if date is True:
        gsv_df = gsv_df.set_index('date')

    return gsv_df

def import_gsv_excel_data(preprocess=True, plot=False):
    gsv_sheets = ['gsv_ticker_2018', 'gsv_ticker_2019', 'gsv_ticker_2020', 'gsv_names_2018', 'gsv_names_2019', 'gsv_names_2020']
    means_df = pd.DataFrame()
    for sheet in gsv_sheets:
        df = import_gsv_sheet(sheet)
        if preprocess is True:
            gsv_data = df.fillna(0)
            index_for_return = gsv_data.columns

            means = gsv_data.mean()
            stds = gsv_data.std()

            means.reset_index(drop=True, inplace=True)

            means_df[sheet] = means

            font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
            font_prop = font_manager.FontProperties(fname=font_path, size=12)

            """
            plt.figure(figsize=(15, 8))
            plt.subplot(1, 2, 1)
#            means.plot(kind='kde', color='grey', lw=2)
            plt.hist(means, bins=65, color='grey', alpha=0.7)#, density=True)
            plt.title(f'Distribution of Means for {sheet}',  fontproperties=font_prop, weight='bold', size=18)
            plt.xlabel('Mean Values of Timeseries', fontproperties=font_prop, weight='bold', size=18)
            plt.ylabel('Density',  fontproperties=font_prop, weight='bold', size=15)

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(rotation=45, ha='right', fontproperties=font_prop)

            plt.subplot(1, 2, 2)
#            stds.plot(kind='kde', color='grey', lw=2)
            plt.hist(stds, bins=65, color='grey', alpha=0.7)#, density=True)
            plt.title(f'Distribution of Standard Deviations for {sheet}',  fontproperties=font_prop, weight='bold', size=18)
            plt.xlabel('Standard Deviations of Timeseries', fontproperties=font_prop, weight='bold', size=18)
            plt.ylabel('Density', fontproperties=font_prop, weight='bold', size=15)

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(rotation=45, ha='right', fontproperties=font_prop)

            plt.tight_layout()
            source_text = 'Source: Refinitiv, 2023'
            plt.figtext(0.98, 0.02, source_text, ha='right', fontproperties=font_prop)

            if plot is True:
                plt.show()
            """

    means_df['Index'] = index_for_return
    means_df = means_df.set_index('Index')

    prefix_mapping = {
        'gsv_ticker_2018': 'GSV Tickers 2018',
        'gsv_ticker_2019': 'GSV Tickers 2019',
        'gsv_ticker_2020': 'GSV Tickers 2020',
        'gsv_names_2018': 'GSV Names 2018',
        'gsv_names_2019': 'GSV Names 2019',
        'gsv_names_2020': 'GSV Names 2020',
    }

    means_df.rename(columns=prefix_mapping, inplace=True)

    return means_df

#gsv_ma_data = import_gsv_excel_data(plot=False)
#print(gsv_ma_data)


"""
def gsv_pre_statistics(df):
    column_medians = df.median()
    column_means = df.mean()
    column_std_devs = df.std()
    column_skew = df.skew()
    column_kurt = df.kurtosis()


    average_std_dev = column_std_devs.mean()

    results_df = pd.DataFrame({
        'Medians': column_medians,
        'Means': column_means,
        'Standard Deviations': column_std_devs,
        'SD-to-Mean': column_means/column_std_devs,
        'Skewnss': column_skew,
        'Kurtosis': column_kurt
    })

    return results_df

# create gsv pre-summary statistics
array_gsv_names = ['pre_ss_ticker_18', 'pre_ss_ticker_19', 'pre_ss_ticker_20', 'pre_ss_names_18', 'pre_ss_names_19', 'pre_ss_names_20', 'pre_ss_names_s_18', 'pre_ss_names_s_19', 'pre_ss_names_s_20']
array_gsv_dfs   = [ticker_18, ticker_19, ticker_20, names_18, names_19, names_20, names_s_18, names_s_19, names_s_20]
#for i in range(0,len(array_gsv_names)):
    #gsv_df = gsv_pre_statistics(df=array_gsv_dfs[i])
    #with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #gsv_df.to_excel(writer, sheet_name=array_gsv_names[i], startrow=0, startcol=0)

ss_ticker_18 = gsv_ma('pre_ss_ticker_18', date=False)
ss_ticker_19 = gsv_ma('pre_ss_ticker_19', date=False)
ss_ticker_20 = gsv_ma('pre_ss_ticker_20', date=False)
ss_names_18 = gsv_ma('pre_ss_names_18', date=False)
ss_names_19 = gsv_ma('pre_ss_names_19', date=False)
ss_names_20 = gsv_ma('pre_ss_names_20', date=False)
ss_names_s_18 = gsv_ma('pre_ss_names_s_18', date=False)
ss_names_s_19 = gsv_ma('pre_ss_names_s_19', date=False)
ss_names_s_20 = gsv_ma('pre_ss_names_s_20', date=False)
array_gsv_pre_ss = [ss_ticker_18, ss_ticker_19, ss_ticker_20, ss_names_18, ss_names_19, ss_names_20, ss_names_s_18, ss_names_s_19, ss_names_s_20]

def calculate_and_add_summary(arr_gsv_pre_ss):
    summary_df = pd.DataFrame(
        columns=['DataFrame', 'Count Zeros Median', 'Difference Median-Mean', 'Average SD-to-Mean', 'Average SD'])

    for i, df in enumerate(arr_gsv_pre_ss):
        count_zeros_median = (df['Medians'] == 0.0).sum()
        difference_median_mean = (df['Medians'] - df['Means']).mean()
        average_sd_to_mean = df['SD-to-Mean'].mean()
        average_sd = df['Standard Deviations'].mean()

        summary_df = summary_df.append({
            'DataFrame': array_gsv_names[i],
            'Count Zeros Median': count_zeros_median,
            'Difference Median-Mean': difference_median_mean,
            'Average SD-to-Mean': average_sd_to_mean,
            'Average SD': average_sd
        }, ignore_index=True)

    with pd.ExcelWriter("GSV_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        summary_df.to_excel(writer, sheet_name='pre_ss_summary', startrow=0, startcol=0)

    return summary_df

#print(calculate_and_add_summary(array_gsv_pre_ss))
"""





