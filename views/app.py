# Lines 4-55 are aadapted and refactored from: https://www.youtube.com/watch?v=oAoSCPu1a3I
# Subjectivity analysis added to the sentiment analysis code: adapted from https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524

# pip install streamlit textblob nltk contractions pyspellchecker vaderSentiment pandas matplotlib openpyxl plotly wordcloud

# Import the required libraries
import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import contractions
from spellchecker import SpellChecker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time


# Load external css file
# Code adapted and modified from: https://medium.com/pythoneers/how-to-customize-css-in-streamlit-a-step-by-step-guide-761375318e05
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom CSS from style.css
local_css('style.css')

html_main = """
    <!-- Main section -->
    <div class="main-section">
        <h1>SentimentPulse</h1>
        <p>Gain a deeper understanding of how your audience feels by analyzing customer feedback, social media comments, and product reviews.</p>
    </div>
"""
html_content = html_main
st.html(html_content)

if 'show_content' not in st.session_state:
    st.session_state.show_content = False

def show_how():
    st.markdown("""
    **1. Collect Data**: Gather customer feedback, social media comments, and product reviews.  
    **2. Analyze Sentiment**: Enter text data into the sentiment analysis tool below and click the 'Analyze' button to assess the emotional tone of the text.  
    **3. Gain Insights**: View the sentiment analysis result to understand audience feelings and identify trends to inform decisions.
    """)

# Create three columns with different widths: 1, 2.8, 1
# Code adapted from: https://discuss.streamlit.io/t/how-to-set-custom-width-for-specific-sections-in-streamlit-app/84288/2
def center_content():
    _, col, _ = st.columns([1, 2.8, 1])
    return col

# Display button and show content when clicked
# Content is displayed in the center column
# Code adapted from: https://docs.streamlit.io/develop/api-reference/widgets/st.button
with center_content() as col:
    if st.button("Show Me How", type="primary"):    
        st.session_state.show_content = not st.session_state.show_content

    if st.session_state.show_content:
        show_how()
        st.markdown("""
            <style>
                .eiemyj0 {
                    height: 220px;
                }
            </style>
        """, unsafe_allow_html=True)

# Download the 'punkt_tab' tokenizer model
nltk.download('punkt_tab')

# Set stopwords for the word cloud
nltk.download('stopwords')
nltk_stopwords = set(stopwords.words('english'))
custom_stopwords = {"experience", "movie", "service", "product", "website", "food", "drink"}
stopwords_combined = nltk_stopwords.union(custom_stopwords)

# Code adapted from https://www.datacamp.com/tutorial/wordcloud-python
# Function to generate word cloud
def generate_wordcloud(text, max_words=25):

    text = preprocess_text(text).lower()
    # Create a WordCloud object
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords=stopwords_combined, 
                    min_font_size = 8,
                    max_words=max_words).generate(text)
    st.image(wordcloud.to_array())

# Code adapted from https://pandas.pydata.org/docs/user_guide/style.html
def style_negative(v, props='color: red'):
    if isinstance(v, (int, float)) and v < 0:
        return props
    else:
        return None

# Coded with help from Claude using idmax() function and pandas Series
def highlight_sentiment_max(row):
    max_col = row[['Positive Score %', 'Neutral Score %', 'Negative Score %']].idxmax()
    
    # Initialize an empty Series with the same index as the row
    styles = pd.Series(index=row.index, data=[''] * len(row))
    
    # Apply different background colors based on which column has max value
    if max_col == 'Positive Score %':
        styles[max_col] = 'background-color: lightgreen'
    elif max_col == 'Neutral Score %':
        styles[max_col] = 'background-color: #FFEB99'  # Light yellow
    elif max_col == 'Negative Score %':
        styles[max_col] = 'background-color: #FFCCCC'  # Light red
    
    return styles

# Preprocessing pipeline - a series of steps that are applied to the raw text before sentiment analysis
# The goal of the preprocessing pipeline is to clean the raw input text and make it suitable for sentiment analysis.
# The preprocessing steps include:
# 1. Contraction expansion 
def expand_contractions(input_text):
    return contractions.fix(input_text)

# 2. Spell checking
def spell_check(input_text):
    spell = SpellChecker() # Create a SpellChecker object
    corrected_word = [spell.correction(word) if spell.correction(word) != word else word for word in input_text.split()] # Correct the spelling of each word
    corrected_word = [word if word is not None else '' for word in corrected_word]
    return corrected_word

# 3. Tokenization - Split the text into words
def preprocess_text(input_text):
    text = expand_contractions(input_text) # Expand contractions
    corrected_word = spell_check(text) # Spell check
    corrected_word = [word for word in corrected_word if word]  # Filter out empty strings
    return " ".join(word_tokenize(" ".join(corrected_word)))

# Function to get sentiment
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer() # Create a SentimentIntensityAnalyzer object

    # Get sentiment scores from VADER for the given text
    # The 'compound' score is a float within the range of [-1.0, 1.0]:
    #   - -1.0 indicates a very negative sentiment
    #   - 1.0 indicates a very positive sentiment
    #   - 0.0 indicates a neutral sentiment
    sentiment_score = analyzer.polarity_scores(text)

    # Extracting individual polarity scores
    # The 'pos', 'neg', and 'neu' scores represent the proportion of text that falls into these categories
    # Modified Vader sentiment analysis code, originally from https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/"
    negative = sentiment_score['neg'] * 100
    neutral = sentiment_score['neu'] * 100
    positive = sentiment_score['pos'] * 100
    compound_score = sentiment_score['compound'] * 100

    # Determine the sentiment type based on the compound score
    # The thresholds used for classification are:
    #   - Positive sentiment if the compound score >= 0.5
    #   - Negative sentiment if the compound score <= -0.5
    #   - Neutral sentiment if the compound score is between -0.5 and 0.5

    if compound_score >= 0.05:
        sentiment_type = "Positive"
    elif compound_score <= -0.05:
        sentiment_type = "Negative"
    else:
        sentiment_type = "Neutral"
    
    # Calculate subjectivity using TextBlob
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity * 100 # Subjectivity score
    
    return sentiment_type, subjectivity, positive, neutral, negative, compound_score


# Get user input
input_text = st.text_area("Ready to Analyze:", placeholder="Type your feedback, comments, or review here...")

# Add a button to perform sentiment analysis
# When the button is clicked, perform sentiment analysis and display the result
button = st.button("Reveal Insights", type="primary", key="insights")

# Initialize the button click state
# If the button is not clicked, set the button_clicked state to False
# Code adapted from: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
# https://www.restack.io/docs/streamlit-knowledge-streamlit-button-color-guide
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Check user input and button click to perform sentiment analysis
# Display the result in the UI using st.write function 
# If no text is provided, display a warning message
# If text is provided, display the sentiment result
if button:
    st.session_state.button_clicked = True

if st.session_state.button_clicked and not input_text:
    st.warning("Please provide some text for analysis.")
elif input_text.strip() != "":
        # Step 1: Preprocess the input text
        input_text = preprocess_text(input_text)
        st.write("Preprocessed Text: ", input_text)

        # Step 2: Perform sentiment analysis using VADER and calculate subjectivity using TextBlob
        sentiment_type, subjectivity, positive, neutral, negative, compound_score = get_sentiment(input_text)

        # Code adapted and modified from: https://docs.streamlit.io/develop/concepts/design/dataframess
        # Create a DataFrame to display the sentiment analysis result
        data = {
            "Sentiment Strength (Compound Score) %": [int(compound_score)],
            "Positive Score %": [int(positive)],
            "Neutral Score %": [int(neutral)],
            "Negative Score %": [int(negative)],
            "Subjectivity Score %": [int(subjectivity)]
        }
        sentiment_data_df = pd.DataFrame(data)
    
        # Apply styling to the Sentiment Type column using lambda function
        # Code adapted from: https://discuss.streamlit.io/t/streamlit-doesnt-seem-to-support-pd-style-applymap-index/31820
        styled_df = sentiment_data_df.style.applymap(lambda x: 'color: #003566')\
            .applymap(style_negative)\
            .applymap(lambda v: 'color: grey' if isinstance(v, (int, float)) and (v < 0.3) and (v > -0.3) else None)\
            .apply(highlight_sentiment_max, axis=1)
        st.dataframe(styled_df, use_container_width=True)

# Code adapted from: https://stackoverflow.com/questions/25962114/how-do-i-read-a-large-csv-file-with-pandas
# Function to load CSV in chunks
def load_csv_chunks(file, chunksize=5):
    file_chunk = pd.read_csv(file, chunksize=chunksize)
    for chunk in file_chunk:
        yield chunk

# Function to load Excel in chunks
# Code fixed and modified with help from ChatGPT
def load_excel_chunks(file, chunksize=5):
    file_chunk = pd.read_excel(file, engine='openpyxl')
    for start_row in range(0, len(file_chunk), chunksize):
        chunk = file_chunk.iloc[start_row:start_row + chunksize]
        yield chunk

def append_sentiment_results(sentiment_results, text, sentiment_type, compound_score, positive, neutral, negative, subjectivity):
    sentiment_results.append({
        "Text": text,
        "Sentiment Type": sentiment_type,
        "Sentiment Strength (Compound Score) %": int(compound_score),
        "Positive Score %": int(positive),
        "Neutral Score %": int(neutral),
        "Negative Score %": int(negative),
        "Subjectivity Score %": int(subjectivity),
    })

# Initialize counters for sentiment types
positive_count = 0
neutral_count = 0
negative_count = 0

# Initialize an empty string to hold all the text
all_text = ""

# Option to upload file (CSV or Excel files)
uploaded_file = st.file_uploader("Or, upload a CSV or Excel file", type=["csv", "xlsx"])

# Check if a file is uploaded
# Code adapted from: https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
if uploaded_file is not None:

    sentiment_results = [] # Initialize an empty list to store sentiment analysis results
    
    # Read the file based on its extension
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'csv':
        # https://docs.streamlit.io/develop/api-reference/status/st.progress
        total_chunks = sum(1 for _ in pd.read_csv(uploaded_file, chunksize=5))  # Total chunks for progress calculation
        uploaded_file.seek(0)
        
        for chunk_index, chunk in enumerate(load_csv_chunks(uploaded_file, chunksize=5)):
            progress_text = f"Processing chunk {chunk_index + 1} of {total_chunks}. Please wait."
            my_bar = st.progress(0, text=progress_text)

            # Update progress bar during chunk processing
            for percent_complete in range(100):
                time.sleep(0.10)
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            for text in chunk['text']:
                preprocessed_text = preprocess_text(text)
                all_text += preprocessed_text + " " # Append the preprocessed text to the all_text string
                sentiment_type, subjectivity, positive, neutral, negative, compound_score = get_sentiment(preprocessed_text)

                # Count the number of sentiment types
                if sentiment_type == 'Positive':
                    positive_count += 1
                elif sentiment_type == 'Neutral':
                    neutral_count +=1
                else:
                    negative_count += 1

                # Append sentiment analysis results
                append_sentiment_results(sentiment_results, text, sentiment_type, compound_score, positive, neutral, negative, subjectivity)
            my_bar.empty()

    elif file_extension == 'xlsx':
        total_chunks = sum(1 for _ in load_excel_chunks(uploaded_file, chunksize=5))  # Get the total number of chunks
        # Refactor progress bar code with help from ChatGPT
    
        for chunk_index, chunk in enumerate(load_excel_chunks(uploaded_file, chunksize=5)):
            progress_text = f"Processing chunk {chunk_index + 1} of {total_chunks} chunks. Please wait."
            my_bar = st.progress(0, text=progress_text)

            # Update progress bar as each chunk is processed
            for percent_complete in range(100):
                time.sleep(0.10)  # Simulating processing time
                my_bar.progress(percent_complete + 1, text=progress_text)

            for text in chunk['text']:
                preprocessed_text = preprocess_text(text)
                all_text += preprocessed_text + " " # Append the preprocessed text to the all_text string
                sentiment_type, subjectivity, positive, neutral, negative, compound_score = get_sentiment(preprocessed_text)

                # Count the number of sentiment types
                if sentiment_type == 'Positive':
                    positive_count += 1
                elif sentiment_type == 'Neutral':
                    neutral_count +=1
                else:
                    negative_count += 1

                # Append sentiment analysis results
                append_sentiment_results(sentiment_results, text, sentiment_type, compound_score, positive, neutral, negative, subjectivity)
            
            my_bar.empty()

    if sentiment_results:
        sentiment_df = pd.DataFrame(sentiment_results)
    
        # Apply styling to the Sentiment Type column using lambda function
        # Code adapted from: https://discuss.streamlit.io/t/streamlit-doesnt-seem-to-support-pd-style-applymap-index/31820
        styled_df = sentiment_df.style.applymap(lambda x: 'color: #003566')\
            .applymap(style_negative)\
            .applymap(lambda v: 'color: grey' if isinstance(v, (int, float)) and (v < 0.3) and (v > -0.3) else None)\
            .apply(highlight_sentiment_max, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Calculate the sentiment distribution based on the sentiment type
        sentiment_distribution = {
            "Positive": positive_count,
            "Neutral": neutral_count,
            "Negative": negative_count
        }

        # Calculate general sentiment type based on the sentiment distribution
        # Refactor code based on https://note.nkmk.me/en/python-dict-value-max-min/
        general_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)

        sentiment_colors = {
            "Positive": "lightgreen",
            "Neutral": "#FFEB99",
            "Negative": "#FFCCCC"
        }
        if general_sentiment in sentiment_colors:
            color = sentiment_colors[general_sentiment]
            st.markdown(
                f"**Overall Sentiment:** <span style='background-color:{color};padding:2px 6px;border-radius:3px'>{general_sentiment}</span>",
                unsafe_allow_html=True
            )
        else:
            # No styling for edge case scenarios
            st.write(f"**Overall Sentiment:** {general_sentiment}")
    
        # Show total number of words in the combined text
        st.write(f"There are {len(all_text.split())} words in the combined text.")
        
        wordcloud_col, pie_col = st.columns(2)     

        with pie_col:
            # Create a pie chart to visualize the sentiment distribution
            # Code adapted from: https://plotly.com/python/pie-charts/ and 
            # https://github.com/nkmk/python-snippets/blob/bf110512d819cd34cdea11dc74cfdc4de9090516/notebook/dict_keys_values_items.py
            # and https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
            # Hover label code adapted from: https://plotly.com/python/hover-text-and-formatting/#customizing-hover-label-appearance

            custom_colors = ["lightgreen", "#FFEB99", "#FFCCCC"]

            sentiment_distribution_pie = px.pie(
                values=sentiment_distribution.values(), 
                names=sentiment_distribution.keys(), 
                color_discrete_sequence=custom_colors, 
                labels={'names': 'Sentiment Type', 'values': 'Count'})
            sentiment_distribution_pie.update_traces(textposition='inside', textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=16,
                    font_family="Montserrat",
                    bordercolor="#003566",
                    )
            )
            sentiment_distribution_pie.update_traces(pull=[0.2, 0.1, 0]) # Pull the slices out
            st.plotly_chart(sentiment_distribution_pie, use_container_width=True)
        
        with wordcloud_col:
            # Generate the word cloud
            generate_wordcloud(all_text)
            
    else:
        st.warning("No valid data found in the uploaded file.")
else:
    st.warning("Please upload a CSV or Excel file to analyze sentiment.")
