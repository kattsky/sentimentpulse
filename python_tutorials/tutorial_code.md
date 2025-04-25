# Quick and Simple Sentiment Analysis App in Python

## Introduction

This file contains the initial code that was used to build the Sentiment Analysis App

## Tutorial Code

```python

# pip install streamlit
# pip install textblob

import streamlit as st
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)

    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

st.title("Sentiment Analysis App")

input = st.text_area("Enter text:")

button = st.button("Analyze")

if button:
    if input:
    sentiment = get_sentiment(input_text)
    st.write("Sentiment:", sentiment)
else:
    st.warning("Please enter some text")
```

## Resource

[Quick and Simple Sentiment Analysis App in Python](https://www.youtube.com/watch?v=oAoSCPu1a3I)
