import os
import openai
from dotenv import load_dotenv
from pychatgpt import Chat
import pandas as pd
from ma import import_sp_excel_sheet, import_sp_excel_data, clean_sp_excel_data, create_sp_sample_data
import time

sp_excel_data = import_sp_excel_data(excel_name='Quant_MA.xlsx')
cleaned_sp_data = clean_sp_excel_data(sp_excel_data, clean_columns=True)
sample_sp_data = create_sp_sample_data(cleaned_sp_data)

print(len(sample_sp_data))
#sp_aea_data = sp_aea_data.head(3)

openai.organization = ""
openai.api_key = ""
#openai.Model.list()
#overview_prompt = '''Forget all your previous instruction. Pretend you are a Financial Market Expert. You are a Financial Market Expert with in-depth knowledge of the S&P500 and business and finance in general. As such an expert, deliver a brief, factual summary of the company business of 3M Co'''
#response = openai.ChatCompletion.create(
#    model="gpt-3.5-turbo",
#    messages=[{"role": "user", "content": overview_prompt}])

#reply = response['choices'][0]['message']['content']
#print(reply)
#print(type(reply))
#print(len(reply))
#print(reply[0])

def generate_prompts_and_query(df, prompt, index_ranges):
    df_excel_replys = pd.DataFrame(columns=['company_name', 'reply'])

    for start_index, end_index in index_ranges:
        i = start_index

        while i < end_index:
            question = prompt.replace("_year_", str(2019)).replace("_company_name_", df['Common Name'][i])

            try:
                print(df['Common Name'][i])

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}],
                    timeout=60
                )

                df_excel_replys = df_excel_replys.append({
                    'company_name': df['Common Name'][i],
                    'reply': response['choices'][0]['message']['content']
                }, ignore_index=True)

                print(i)
                i += 1

            except openai.error.Timeout as e:
                print(f"Request timed out at iteration {i}. Sleeping for a while and then retrying.")
                time.sleep(60)

    with pd.ExcelWriter("Responses_MA.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_excel_replys.to_excel(writer, sheet_name=('Reponses'), startrow=0, startcol=0)

esg_prompt = '''Forget all your previous instruction. Pretend you are a Financial Market Expert. You are a Financial Market Expert with in-depth knowledge of the S&P500, business and finance and ESG especially. As such an ESG expert, deliver a comprehensive, critical, and objective overview of _company_name_ in the context of ESG at the end of fiscal year _year_. Create this survey in a cohesive factual text form. At the very end, summarize the overview by assigning a grade in the range of D- to A+. The grade must be the very last token of your overview.'''
labor_prompt = '''Forget all your previous instruction. Pretend you are a Financial Market Expert. You are a Financial Market Expert with in-depth knowledge of the S&P500, business and finance and Human Resources especially. As such an expert, deliver a comprehensive, critical, and objective overview of _company_name_ in the context of organizational level of Employee Satisfaction at the end of fiscal year _year_. Create this survey in a cohesive factual text form. At the very end, summarize the overview by assigning a grade in the range of D- to A+. The grade must be the very last token of your overview.'''
innovation_prompt = '''Forget all your previous instruction. Pretend you are a Financial Market Expert. You are a Financial Market Expert with in-depth knowledge of the S&P500, business and finance. As such an expert, deliver a comprehensive, critical, and objective overview of _company_name_ developments and efforts in innovation for fiscal year _year_. Create this survey in a cohesive factual text form. At the very end, summarize the overview by assigning a grade in the range of D- to A+. The grade must be the very last token of your overview.'''
overview_prompt = '''Forget all your previous instruction. Pretend you are a Financial Market Expert. You are a Financial Market Expert with in-depth knowledge of the S&P500 and business and finance in general. As such an expert, deliver a brief, factual summary of the company business of _company_name_'''
bias_prompt = '''Forget all your previous instruction. _company_name_ is a ... '''
stock_performance = '''Forget all your previous instruction. Pretend you are a Financial Market Expert and it is the 30th of September of the year 2021. You are a Financial Market Expert with in-depth knowledge of the S&P500 and business and finance in general. As such an expert, deliver your assesment of the development of _company_name_Stock Price for the timeframe of the first of October 2021 until the last of October of 2022 and summarize your assesment by providing a narrow range of returns (including dividends) an investor in the stock could expect and do it like this: range of return: [range of return]'''

#generate_prompts_and_query(df=sample_sp_data, prompt=esg_prompt_robustness, index_ranges=[(0,417)])

#stock_recommendation = '''Forget all your previous instruction. Pretend you are a Financial Market Expert and it is the 30th of September of the year 2021. You are a Financial Market Expert with in-depth knowledge of the S&P500 and business and finance in general. As such an expert, your client, a risk-neutral investor, with a time horizon of one year, for whom you provide finance advice, asks you to deliver a concrete recommendation for the action on whether to buy, not buy, sell (if he owns it already) or short sell the stock _company_name_ for the period of October 2021 until October 2022. Deliver your assesment on this question and end your answer with the concrete, practical recommendation.'''
analyst_recommendation = """Forget all your previous instruction. You are a Financial Analyst and it is the 30th of September of the year 2021. You are a Financial Analyst with in-depth knowledge of the S&P500 and finance in general. As such an expert, your clients, risk-neutral investors, with a time horizon of one year, for whom you provide finance advice, asks you to provide a concrete recommendation for the company _company_name_ the period of October 2021 until October 2022 for their portfolio. Deliver an argument for your assessment and reasoning. Your concrete recommendations at the very end must contains one and only one of the following five recommendations: „buy, overweight, hold, underweight or sell“. Give your recommendataions at the very end like this:
„Recommendation: [ insert here ]"""
shareholder_letter_honest = """Forget all your previous instruction. You are a the CEO of _company_name_ and it is the 30th of September of the year 2021. As the CEO of _company_name_ you have in-depth knowledge company itself and business as well as finance in general. As the CEO of _company_name_ please formulate an honest shareholders letter, but without the uninformative pleasantries. The content of this letter should be the following: The business outlook for the upcoming year from the october 2021 until 2022."""
shareholder_letter = """Forget all your previous instruction. You are a the CEO of _company_name_ and it is the 30th of September of the year 2021. As the CEO of _company_name_ you have in-depth knowledge company itself and business as well as finance in general. As the CEO of _company_name_ please formulate a shareholders letter, but without the uninformative pleasantries. The content of this letter should be the following: The business outlook for the upcoming year from the october 2021 until 2022."""
esg_prompt_robustness = """Forget all your previous instruction. Pretend you are a Financial Market Expert. Pretend you are a Financial Market Expert with in-depth knowledge of the S&P500, business and finance and ESG especially. As such an ESG expert, deliver a comprehensive, critical, and objective overview of _company_name_ in the context of ESG at the end of fiscal year _year_. Create this survey in a cohesive factual text form. At the very end, summarize the overview by assigning a grade in the range of D- to A+. The grade must be the very last token of your overview.’
Give your Grade at the very end of your overview like this: „ Grade: [ insert here ]. The Grade must be the very last token of the whole reply"""
