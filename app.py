import streamlit as st
from extract_info import extract_chain, highlight_text, get_unbias_suggestion, calculate_overall_unbias_score, calculate_bias_percentage, calculate_average_sentence_length, \
    create_bar_plot, create_pie_chart, generate_word_cloud, create_sentence_length_histogram, set_openai_api_key
from langchain.document_loaders import PyPDFLoader
from default_text import default_text3
from streamlit_option_menu import option_menu
import tempfile
from markup import app_intro, how_use_intro


def tab1():

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image.png", use_column_width=True)
    with col2:
        st.markdown(app_intro(), unsafe_allow_html=True)
    st.markdown(how_use_intro(),unsafe_allow_html=True) 


    github_link = '[<img src="https://badgen.net/badge/icon/github?icon=github&label">](https://github.com/ethanrom)'
    huggingface_link = '[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/ethanrom)'

    st.write(github_link + '&nbsp;&nbsp;&nbsp;' + huggingface_link, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 14px; color: #777;'>Disclaimer: This app is a proof-of-concept and may not be suitable for real-world legal or policy decisions.</p>", unsafe_allow_html=True)

def tab2():

    openai_api_key = st.text_input("Enter your OpenAI API key:", type='password')

    st.title("🚫 Textual Bias Removal and Mitigation in Policies")
    st.markdown("<p style='font-size: 20px; color: #0080ff; text-align: center;'>Identify and Mitigate Biases in Policy Text</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("🔍 Bias Removal")
    st.write("Uncover and address biases in your policy text to promote fairness and inclusivity.")

    input_option = st.radio("Choose input option:", ("📝 Enter Text Manually", "📄 Upload a PDF File"))

    if input_option == "📄 Upload a PDF File":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            with st.spinner("Extracting information..."):
                input_text = ""
                loader = PyPDFLoader(temp_file.name)
                pages = loader.load_and_split()

                for page in pages:
                    input_text += page.page_content
    else:
        input_text = st.text_area("Enter your text:", height=400, value=default_text3)

    if st.button("Analyze Document"):
        if openai_api_key:
            set_openai_api_key(openai_api_key)

            with st.spinner("Searching for biases..."):
                info = extract_chain.run(input_text)

                gender_bias_sentence_list = []
                race_bias_sentence_list = []
                age_bias_sentence_list = []

                for item in info:
                    if item['gender_bias_sentence']:
                        gender_bias_sentence_list.append(item['gender_bias_sentence'])
                    if item['racial_bias_sentence']:
                        race_bias_sentence_list.append(item['racial_bias_sentence'])
                    if item['age_bias_sentence']:
                        age_bias_sentence_list.append(item['age_bias_sentence'])

                highlighted_text = input_text

                for gender_bias_sentence in gender_bias_sentence_list:
                    highlighted_text = highlighted_text.replace(gender_bias_sentence, highlight_text(gender_bias_sentence, 'yellow'))

                for racial_bias_sentence in race_bias_sentence_list:
                    highlighted_text = highlighted_text.replace(racial_bias_sentence, highlight_text(racial_bias_sentence, 'orange'))

                for age_bias_sentence in age_bias_sentence_list:
                    highlighted_text = highlighted_text.replace(age_bias_sentence, highlight_text(age_bias_sentence, 'red'))

                st.subheader("Results")

                with st.expander("View Highlighted Document"):
                    st.markdown(f"<div>{highlighted_text}</div>", unsafe_allow_html=True)

                overall_unbias_score = calculate_overall_unbias_score(input_text, gender_bias_sentence_list, race_bias_sentence_list, age_bias_sentence_list)

                st.markdown("<h1 style='text-align: center; color: #0080ff;'>🌐 Overall Unbiasedness Score</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-size: 18px;'>Calculated by comparing total sentences and bias sentences</p>", unsafe_allow_html=True)

                if overall_unbias_score == 100:
                    score_style = "color: green; font-size: 2em;"
                elif overall_unbias_score > 90:
                    score_style = "color: green; font-size: 2em;"
                elif overall_unbias_score >= 50:
                    score_style = "color: orange; font-size: 2em;"
                else:
                    score_style = "color: red; font-size: 2em;"

                score_html = f"<span style='{score_style}'>{overall_unbias_score:.2f}%</span>"
                centered_html = f"<div style='text-align: center;'>{score_html}</div>"
                st.write(centered_html, unsafe_allow_html=True)

                st.markdown("<h2 style='color: #ff5733;'>🚨 Detected Bias Sentences</h2>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 16px;'>These are the sentences we detected as potentially biased and in need of improvement.</p>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Gender Bias Sentences")
                    st.write(gender_bias_sentence_list)

                with col2:
                    st.subheader("Racial Bias Sentences")
                    st.write(race_bias_sentence_list)

                with col3:
                    st.subheader("Age Bias Sentences")
                    st.write(age_bias_sentence_list)

                unbiased_suggestions = []
                for bias_sentence in gender_bias_sentence_list + race_bias_sentence_list + age_bias_sentence_list:
                    unbiased_sentence = get_unbias_suggestion(bias_sentence)
                    unbiased_suggestions.append(unbiased_sentence)

                if unbiased_suggestions:
                    st.markdown("<h2 style='color: #008000;'>📝 Rewritten Sentences for Possible Bias</h2>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 16px;'>These are the improved versions of the potentially biased sentences.</p>", unsafe_allow_html=True)
                    st.write(unbiased_suggestions)

                with st.expander("Other Analytics"):
                    sentences = input_text.split('\n')

                    gender_bias_percentage = calculate_bias_percentage(sentences, gender_bias_sentence_list) if gender_bias_sentence_list else 0
                    race_bias_percentage = calculate_bias_percentage(sentences, race_bias_sentence_list) if race_bias_sentence_list else 0
                    age_bias_percentage = calculate_bias_percentage(sentences, age_bias_sentence_list) if age_bias_sentence_list else 0

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Gender Bias Percentage: {gender_bias_percentage}")

                    with col2:    
                        st.write(f"Race Bias Percentage: {race_bias_percentage}")
                    
                    with col3:
                        st.write(f"Age Bias Percentage: {age_bias_percentage}")

                    average_sentence_length = calculate_average_sentence_length(sentences)
                    st.write(f"Average Sentence Length: {average_sentence_length}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        bias_labels = ["Gender Bias", "Race Bias", "Age Bias"]
                        bias_counts = [gender_bias_percentage, race_bias_percentage, age_bias_percentage]
                        create_bar_plot(bias_labels, bias_counts)

                    with col2:
                        bias_percentages = [gender_bias_percentage, race_bias_percentage, age_bias_percentage]
                        create_pie_chart(bias_labels, bias_percentages)

                    with col3:
                        create_sentence_length_histogram(sentences)

                with st.expander("Word Cloud"):
                    st.subheader("Word Cloud")
                    generate_word_cloud(sentences)
            
        else:
            st.warning("Please Enter Your Openai API Key First")


def main():
    st.set_page_config(page_title="Bias Detection", page_icon=":memo:", layout="wide")
    tabs = ["Introduction", "Bias Detection and Mitigation"]
    
    with st.sidebar:

        current_tab = option_menu("Select a Tab", tabs, menu_icon="cast")
    
    tab_functions = {
    "Introduction": tab1,
    "Bias Detection and Mitigation": tab2,
    }

    if current_tab in tab_functions:
        tab_functions[current_tab]()

if __name__ == "__main__":
    main()