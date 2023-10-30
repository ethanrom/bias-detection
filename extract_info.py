from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import streamlit as st
import openai

def set_openai_api_key(api_key):
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key

short_llm = ChatOpenAI()
long_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

schema = {
    "properties": {
        "gender_bias_sentence": {"type": "string"},
        "racial_bias_sentence": {"type": "string"},
        "age_bias_sentence": {"type": "string"},
    },
    "required": [],
}

extract_chain = create_extraction_chain(schema, long_llm)

def highlight_text(text, color):
    return f"<span style='background-color: {color};'>{text}</span>"


remove_biases_template = """Given a sentence thats containing a gender, age or racial bias, provide an unbiased version of the sentence.

Bias Sentence:{bias_sentence}
Unbiased Sentence:"""

input_variables=["bias_sentence"]

def get_unbias_suggestion(bias_sentence):

    remove_biases_prompt = PromptTemplate(template=remove_biases_template, 
                            input_variables=input_variables,
                            )

    remove_biases_llmchain = LLMChain(prompt=remove_biases_prompt, llm=short_llm)

    filtered_data = remove_biases_llmchain.run(bias_sentence=bias_sentence)
    return filtered_data

def calculate_overall_unbias_score(input_text, gender_bias_sentence_list, race_bias_sentence_list, age_bias_sentence_list):
    sentences = input_text.split('\n')
    total_sentences = len(sentences)
    total_bias_sentences = 0

    for sentence in sentences:
        is_gender_bias = any(gender_bias_sentence in sentence for gender_bias_sentence in gender_bias_sentence_list)
        is_race_bias = any(race_bias_sentence in sentence for race_bias_sentence in race_bias_sentence_list)
        is_age_bias = any(age_bias_sentence in sentence for age_bias_sentence in age_bias_sentence_list)

        if is_gender_bias or is_race_bias or is_age_bias:
            total_bias_sentences += 1

    overall_unbias_score = (total_sentences - total_bias_sentences)*100 / total_sentences
    return overall_unbias_score


def calculate_bias_percentage(sentences, bias_sentence_list):
    total_sentences = len(sentences)
    total_bias_sentences = sum(any(bias_sentence in sentence for bias_sentence in bias_sentence_list) for sentence in sentences)
    bias_percentage = (total_bias_sentences / total_sentences) * 100
    return bias_percentage

def calculate_average_sentence_length(sentences):
    total_characters = sum(len(sentence) for sentence in sentences)
    average_sentence_length = total_characters / len(sentences)
    return average_sentence_length

def create_bar_plot(bias_labels, bias_counts):
    fig, ax = plt.subplots()
    ax.bar(bias_labels, bias_counts)
    ax.set_xlabel('Bias Type')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bias Sentences')
    st.pyplot(fig)

def create_pie_chart(bias_labels, bias_percentages):
    fig, ax = plt.subplots()
    ax.pie(bias_percentages, labels=bias_labels, autopct='%1.1f%%')
    ax.set_title('Percentage of Bias Sentences')
    st.pyplot(fig)

def generate_word_cloud(sentences):
    word_cloud_text = ' '.join(sentences)
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate(word_cloud_text)
    fig, ax = plt.subplots()
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def create_sentence_length_histogram(sentences):
    sentence_lengths = [len(sentence) for sentence in sentences]
    fig, ax = plt.subplots()
    ax.hist(sentence_lengths, bins='auto', alpha=0.7)
    ax.set_xlabel('Sentence Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sentence Lengths')
    st.pyplot(fig)