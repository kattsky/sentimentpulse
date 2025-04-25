# Lines 4-55 are aadapted and refactored from: https://www.youtube.com/watch?v=oAoSCPu1a3I
# Subjectivity analysis added to the sentiment analysis code: adapted from https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524

# pip install streamlit
# pip install textblob
# pip install nltk

# Import the required libraries
import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize

# Download the 'punkt' tokenizer model
nltk.download('punkt')

# Function to get sentiment
def get_sentiment(text):
    blob = TextBlob(text)

    # Get the sentiment score
    # The sentiment score is a float within the range [-1.0, 1.0]
    # -1.0 indicates a very negative sentiment
    # 1.0 indicates a very positive sentiment
    # 0.0 indicates a neutral sentiment
    sentiment_score = blob.sentiment.polarity
    sentiment_percentage = sentiment_score * 100
    subjectivity = blob.sentiment.subjectivity * 100 # Subjectivity score

    if sentiment_score > 0:
        sentiment_type = "Positive"
    elif sentiment_score < 0:
        sentiment_type = "Negative"
    else:
        sentiment_type = "Neutral"
    
    return sentiment_type, sentiment_percentage, subjectivity

# Set the title of the app
st.title("SentimentPulse")

# Get user input
input_text = st.text_area("Enter text:")

# Add a button to perform sentiment analysis
# When the button is clicked, perform sentiment analysis and display the result
button = st.button("Analyze")

# Check user input and button click to perform sentiment analysis
# Display the result in the UI using st.write function 
# If no text is provided, display a warning message
# If text is provided, display the sentiment result
if button and input_text.strip() != "":
    sentiment_type, sentiment_percentage, subjectivity = get_sentiment(input_text)
    st.write(f"Sentiment: {sentiment_type}")

    # Display the sentiment and subjectivity score
    st.write(f"Sentiment Strength Score: {int(sentiment_percentage)}%")
    st.write(f"Subjectivity Score: {int(subjectivity)}%")
else:
    st.warning("Please provide some text for analysis.")