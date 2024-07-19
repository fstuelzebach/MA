# Master Thesis Repository: Uncovering ChatGPT Biases in S&P 500 Company Responses

This repository contains all the materials related to the research conducted for the Master Thesis titled:
_"Uncovering ChatGPT Biases in S&P 500 Company Responses: A Qualitative and Quantitative Analysis"_

Here you will find files, documents, and Python code used in the study. A short summary of the paper is provided at the bottom. 

The thesis was completed under the supervision of Prof. Dr. Gregor Weiß at the University of Leipzig, Chair of Business Administration/Sustainable Financial Services, specifically Banking, received a grade of 1.0 and can be distributed bilaterally if interest exist.

## Contents

### Data Files
- `Quant_MA.xlsx`: Contains the quantitative input data used in the thesis (Refintiv EIKON).
- `BWCount_MA.xlsx`: Data related to word counts.
- `Further Findings.xlsx`: Additional findings and data.
- `GSV_MA.xlsx`: Data related to GSV (Google Search Volume).
- `Grades_MA.xlsx`: Data on grading and evaluation (provided by ChatGPT).
- `Responses_MA.xlsx`: Outputs from ChatGPT interactions (the AI's unfiltered responses).
- `Sentiment_MA.xlsx`: Sentiment analysis results based on the responses.

### Dictionaries
- `LM_Dict.xlsx`: Dictionary used for sentiment analysis (Loughran and McDonald).
- `H4_Dict.xlsx`: Dictionary for additional qualitative analysis (Harvard IV).
- `SWN3.0_Dict.xlsx`: Sentiment WordNet 3.0 dictionary.
- `VADER_Dict.xlsx`: VADER (Valence Aware Dictionary and Sentiment Reasoner) dictionary.

### Python Code
- `ma.py`: Main Python script containing the core functionality of the analysis.
- `dict_ma.py`: Python script for handling dictionary-related functions.
- `gsv_ma.py`: Python script for processing GSV-related data.
- `ChatGPT_API.py`: Python script for interacting with the ChatGPT API.

### Prompts
- `Prompts_MA.docx`: Document listing the specific prompts used in the research (and more).

## Summary

This thesis investigates the biases present in ChatGPT's responses, particularly within the financial domain of S&P 500 companies. ChatGPT, an advanced AI developed by OpenAI, processes vast amounts of textual data to generate human-like responses. Despite its sophisticated design and extensive training, the potential for biases in its outputs exists due to the nature of its training data and underlying algorithms.

The study aims to uncover whether ChatGPT's responses exhibit systematic biases—deviations from objective facts or subjective distortions—when addressing inquiries about S&P 500 companies. The research employs a dual methodology combining quantitative and qualitative approaches to analyze these biases.

Quantitatively, the study develops a novel approach to measure deviations in ChatGPT's company-specific evaluations by comparing them with objective benchmarks. Qualitatively, sentiment analysis is used to examine the tone and potential biases in the language used in ChatGPT's responses.

The findings reveal that while many responses align with expected neutrality, notable biases are observed, especially for companies with significant media presence or those in controversial industries. This work not only highlights potential distortions in ChatGPT's financial assessments but also provides a foundation for future research on biases in AI models and their implications for financial analysis and decision-making.
