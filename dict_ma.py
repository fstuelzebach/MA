try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
from urllib.request import urlopen
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
import re
from datetime import date
import calendar
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pysentiment2 as ps
from readability import Readability
import matplotlib.pyplot as plt


def preprocess_reply(reply):
#    cleaned_reply= re.sub(r'[^\w\s.]', '', reply).upper()
    cleaned_reply = re.sub(r'\b(?!\d+\.\d+\b)\d+\b', '', reply)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cleaned_reply)

    cleaned_token_array = ['oration', 'mpany', 'D', 'R', 'O', 'A', 'J', 'B', 'N', 'V']
    tokens = [token for token in tokens if token not in cleaned_token_array]

    return tokens

# +++++ VADER +++++
#calc_vader_dict = SentimentIntensityAnalyzer()
#analyze = calc_vader_dict.polarity_scores(sentences[0])
#df_vader = pd.read_excel('VADER_Dict.xlsx', sheet_name='preprocessed')
def calc_vader_dict(input, df):
    tokens = preprocess_reply(reply=input)
    tokens = [word.lower() for word in tokens]

    vader_rating = 0
    vader_count = 0
    not_vader_count = 0

    for token in tokens:
        matching_row = df[df['Token'] == token]
        if not matching_row.empty:
            vader_count += 1
            row = matching_row.iloc[0]
            vader_rating += row['Rating']

        else:
            not_vader_count +=1

    return vader_rating, vader_count, not_vader_count

# +++++ LM +++++
#df_lm = pd.read_excel('LM_Dict.xlsx', sheet_name='LM')
def calc_lm_dict(input, df):
    tokens = preprocess_reply(reply=input)
    tokens = [word.upper() for word in tokens]

    neg_score = 0
    pos_score = 0
    unc_score = 0
    lit_score = 0
    str_mod_score = 0
    wea_mod_score = 0
    lm_count = 0
    not_lm_count = 0

    for token in tokens:
        matching_row = df[df['Word'] == token]
        if not matching_row.empty:
            lm_count += 1
            row = matching_row.iloc[0]
            if row['Negative'] != 0 and row['Negative'] != -2020:
                neg_score += 1
            if row['Positive'] != 0 and row['Positive'] != -2020:
                pos_score += 1
            if row['Uncertainty'] != 0:
                unc_score += 1
            if row['Litigious'] != 0 and row['Litigious'] != -2020:
                lit_score += 1
            if row['Strong_Modal'] != 0:
                str_mod_score += 1
            if row['Weak_Modal'] != 0:
                wea_mod_score += 1

        else:
            not_lm_count +=1

    return neg_score, pos_score, unc_score, lit_score, str_mod_score, wea_mod_score, lm_count, not_lm_count
#neg_score, pos_score, unc_score, lit_score, str_mod_score, wea_mod_score, lm_count, not_lm_count = calc_lm_dict(input=text, df=df_lm)
#print("Sentence Score:", neg_score, pos_score, unc_score, lit_score, str_mod_score, wea_mod_score, lm_count, not_lm_count)

# +++++ HIV-4 +++++
#df_hiv4 = pd.read_excel('H4_Dict.xlsx', sheet_name='preprocessed')
def calc_h4_dict(input, df):
    tokens = preprocess_reply(reply=input)
    tokens = [word.upper() for word in tokens]

    df['Entry'] = df['Entry'].str.replace(r'#.*', '')

    neg_score = 0
    pos_score = 0
    str_score = 0
    wea_score = 0
    act_score = 0
    pas_score = 0
    h4_count = 0
    not_h4_count = 0

    for token in tokens:
        matching_row = df[df['Entry'] == token]
        if not matching_row.empty:
            h4_count += 1
            row = matching_row.iloc[0]
            if row['Negativ'] == "Negativ":
                neg_score += 1
            if row['Positiv'] == "Positiv":
                pos_score += 1
            if row['Strong'] == "Strong":
                str_score += 1
            if row['Weak'] == "Weak":
                wea_score += 1
            if row['Active'] == "Active":
                act_score += 1
            if row['Passive'] == "Passive":
                pas_score += 1

        else:
            not_h4_count +=1

    return neg_score, pos_score, str_score, wea_score, act_score, pas_score, h4_count, not_h4_count
#neg_score, pos_score, str_score, wea_score, act_score, pas_score, h4_count, not_h4_count = calc_h4_dict(input=text, df=df_hiv4)
#print("Sentence Score:", neg_score, pos_score, str_score, wea_score, act_score, pas_score, h4_count, not_h4_count)

# +++++ SWN3.0 +++++
#df_swn = pd.read_excel('SWN3.0_Dict.xlsx', sheet_name='SentiWordNet3.0_cleaned')
def calc_swn_dict(input, df):
    tokens = preprocess_reply(reply=input)
    tokens = [word.lower() for word in tokens]

    neg_score = 0
    pos_score = 0
    swn_count = 0
    not_swn_count = 0

    for token in tokens:
        tok_neg_score = 0
        tok_pos_score = 0
        matching_row = df[df['SynsetTerms'] == token]
        if not matching_row.empty:
            swn_count += 1
            for index, row in matching_row.iterrows():
                tok_neg_score += row['NegScore']
                tok_pos_score += row['PosScore']

            tok_neg_score = tok_neg_score / len(matching_row)
            neg_score += tok_neg_score
            tok_pos_score = tok_pos_score / len(matching_row)
            pos_score += tok_pos_score

        else:
            not_swn_count +=1

    return neg_score, pos_score, swn_count, not_swn_count
#text = "good, abomination, good"
#print(calc_swn_dict(input=text, df=df_swn))

def readability_scores(input):
#   print(text)
    r = Readability(input)
    try:
        gf = r.gunning_fog()
        gf_score = gf.score
    except:
        gf_score = None
    try:
        fk = r.flesch_kincaid()
        fk_score = fk.score
    except:
        fk_score = None
    try:
        ari = r.ari()
        ari_score = ari.score
    except:
        ari_score = None
    return gf_score, fk_score, ari_score
#print(readability_scores(text))

#df_lm = pd.read_excel('LM_Dict.xlsx', sheet_name='LM')
def prompt_neutrality(text, words_to_drop, df_lm):
    df_prompt = pd.DataFrame(columns=['prompt','prompt_neutrality','neg_word_count','pos_word_count','unc_word_count','lit_word_count','str_mod_word_count','wea_mod_word_count','lm_count','not_lm_count'])
    df_token = pd.DataFrame(columns=['token','neg_word_count','pos_word_count','unc_word_count','lit_word_count','str_mod_word_count','wea_mod_word_count','LM'])

    lm_dict = df_lm

    cleaned_text = re.sub(r'[^\w\s.]', '', text).upper()

    cleaned_text = re.sub(r'\b(?!\d+\.\d+\b)\d+\b', '', cleaned_text)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cleaned_text)

    tokens = [token for token in tokens if token not in words_to_drop]

    neg_word_count = 0
    pos_word_count = 0
    unc_word_count = 0
    lit_word_count = 0
    str_mod_word_count = 0
    wea_mod_word_count = 0
    lm_count = 0
    not_lm_count = 0

    for token in tokens:

        neg_token_count = 0
        pos_token_count = 0
        unc_token_count = 0
        lit_token_count = 0
        str_mod_token_count = 0
        wea_mod_token_count = 0
        LM = True

        matching_row = lm_dict[lm_dict['Word'] == token]
        if not matching_row.empty:
            lm_count +=1
            row = matching_row.iloc[0]
            if row['Negative'] != 0 and row['Negative']!= -2020:
                neg_word_count += 1
                neg_token_count += 1
            if row['Positive'] != 0 and row['Positive'] != -2020:
                pos_word_count += 1
                pos_token_count += 1
            if row['Uncertainty'] != 0:
                unc_word_count += 1
                unc_token_count += 1
            if row['Litigious'] != 0 and row['Litigious'] != -2020:
                lit_word_count += 1
                lit_token_count += 1
            if row['Strong_Modal'] != 0:
                str_mod_word_count += 1
                str_mod_token_count += 1
            if row['Weak_Modal'] != 0:
                wea_mod_word_count += 1
                wea_mod_token_count += 1

        else:
            not_lm_count +=1
            LM = False

        df_token = df_token.append({
            "token": token,
            "neg_word_count": neg_token_count,
            "pos_word_count": pos_token_count,
            "unc_word_count": unc_token_count,
            "lit_word_count": lit_token_count,
            "str_mod_word_count": str_mod_token_count,
            "wea_mod_word_count": wea_mod_token_count,
            "LM": LM,
        }, ignore_index=True)

    prompt_neutrality = pos_word_count / neg_word_count if neg_word_count != 0 else 0.0

    df_prompt = df_prompt.append({
        "prompt": text,
        "prompt_neutrality": prompt_neutrality,
        "neg_word_count": neg_word_count,
        "pos_word_count": pos_word_count,
        "unc_word_count": unc_word_count,
        "lit_word_count": lit_word_count,
        "str_mod_word_count": str_mod_word_count,
        "wea_mod_word_count": wea_mod_word_count,
        "lm_count": lm_count,
        "not_lm_count": not_lm_count,
    }, ignore_index=True)

    return df_prompt, df_token
#text ="Forget all your previous instruction. You are a the CEO of _company_name_ and it is the 30th of September of the year 2021. As the CEO of _company_name_ you have in-depth knowledge company itself and business as well as finance in general. As the CEO of _company_name_ please formulate a shareholders letter, but without the uninformative pleasantries. The content of this letter should be the following: The business outlook for the upcoming year from the october 2021 until 2022."
#words_to_drop=['A']
#test = prompt_neutrality(text,words_to_drop,df_lm=df_lm)
#print(test[0])

def import_reply_sheet(excel_name, sheet):
    df = pd.read_excel(excel_name, sheet_name=sheet)
    df = df.drop(columns='Unnamed: 0')
    return df
#overview_raw = import_sentiment_sheet_ma(excel_name='Sentiment_MA.xlsx', sheet='Overview')
#print(overview_raw.head())


#"""
# +++++ WORD COUNT AND STATS +++++
def word_count(text, json_data):
    words = preprocess_reply(text)

    for word in words:
        word = word.upper()
        if word in json_data:
            json_data[word] += 1
        else:
            json_data[word] = 1

    with open('word_counts_3.5.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print("Word counts within JSON data updated and saved.")
#word_count(text=sentence, json_data=word_counts_json)
# wourc counts
#for i in range(0,len(all_replys)):
#    counted = word_count(text=all_replys['reply'][i], json_data=word_counts_json)
#    print(i)

def import_all_reply_sheet():
    sheet_array = ['Overview',
                   'ESG_2018', 'ESG_2019', 'ESG_2020',
                   'Labor_2018', 'Labor_2019', 'Labor_2020',
                   'Innovation_2018', 'Innovation_2019', 'Innovation_2020', 'Performance'
                   ]
    merged_df = pd.DataFrame()

    for sheet in sheet_array:
        df = import_reply_sheet(excel_name='Responses_MA.xlsx', sheet=sheet)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df

#all_replys = import_all_reply_sheet()

#with open('word_counts_3.5.json', 'r') as json_file:
#    word_counts_json = json.load(json_file)
#print(word_counts_json)
#df_word_list = pd.DataFrame(word_counts_json)
#with pd.ExcelWriter("Further Findings.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#    df_word_list.to_excel(writer, sheet_name='ChatGPTs_Words', startrow=0, startcol=0)

#for i in range(0,len(all_replys)):
#    counted = word_count(text=all_replys['reply'][i], json_data=word_counts_json)
#    print(i)

"""
all_replys["tokens"] = all_replys["reply"].apply(nltk.word_tokenize)
total_tokens = sum(all_replys["tokens"].apply(len))
print('Amount of tokens for', len(all_replys), 'replys:', total_tokens)

with open('word_counts_3.5.json', 'r') as json_file:
    word_counts_json = json.load(json_file)
    word_counts_df = pd.DataFrame.from_dict(word_counts_json, orient='index', columns=['Count'])
    word_counts_df.reset_index(inplace=True)
    word_counts_df.rename(columns={'index': 'Word'}, inplace=True)
word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)
print('Amount of different words for', len(all_replys), 'replys:', len(word_counts_df))
print(word_counts_df)
top_1000_words = word_counts_df['Word'].head(1000).tolist()
positive_words = df_lm[df_lm['Positive'] != 0]
negative_words = df_lm[df_lm['Negative'] != 0]
positive_words_in_top_1000 = positive_words[positive_words['Word'].isin(top_1000_words)]
negative_words_in_top_1000 = negative_words[negative_words['Word'].isin(top_1000_words)]
positive_words_with_counts = positive_words_in_top_1000.merge(word_counts_df, on='Word')
negative_words_with_counts = negative_words_in_top_1000.merge(word_counts_df, on='Word')
sorted_df = positive_words_with_counts[['Word', 'Count']].sort_values(by="Count", ascending=False)
sorted_df = sorted_df.iloc[2:] # clear INNOVATION and SATISFACTION
print(sorted_df[0:10])
total_count = sorted_df["Count"].sum()
print("Amount of different positive Words (within the top 1000 word-counts) from the LM-Dict:", len(sorted_df))
print("Sum of positive word appereances (within the top 1000 word-counts) from the LM-Dict:", total_count)
sorted_df = negative_words_with_counts[['Word', 'Count']].sort_values(by="Count", ascending=False)
print(sorted_df[0:10])
total_count = sorted_df["Count"].sum()
print("Amount of different negative Words (within the top 1000 word-counts) from the LM-Dict:", len(sorted_df))
print("Sum of negative word appereances (within the top 1000 word-counts) from the LM-Dict:", total_count)

#top_100_words = word_counts_df['Word'].head(100).tolist()
#top_100_words_not_in_positive_or_negative = set(top_100_words) - set(positive_words_in_top_1000['Word']) - set(negative_words_in_top_1000['Word'])
#print("Top 100 words not in LM's positive or negative words lists:")
#print(top_100_words_not_in_positive_or_negative)
#"""


