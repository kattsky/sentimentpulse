import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import streamlit as st
from io import StringIO
from unittest.mock import patch
import numpy as np
from wordcloud import WordCloud
from views.app import expand_contractions, spell_check, preprocess_text, get_sentiment, generate_wordcloud

def test_expand_contractions():
    input_text = "I'm not sure how it'll work."
    expected_output = "I am not sure how it will work."
    assert expand_contractions(input_text) == expected_output

def test_spell_check_misspelled_words():
    input_text = "I lov this prodoct!"
    expected_output = ["I", "love", "this", "product"]
    assert spell_check(input_text) == expected_output

def test_preprocess_text():
    input_text = "I'm lovin' this prduct! It’s great."
    expected_output = "I am loving this product It is great"
    assert preprocess_text(input_text) == expected_output

def test_get_sentiment_positive():
    text = "I absolutely love this product!"
    sentiment_type, subjectivity, positive, neutral, negative, compound_score = get_sentiment(text)
    assert sentiment_type == "Positive"
    assert compound_score >= 0.05
    assert positive > neutral and positive > negative

def test_get_sentiment_negative():
    text = "I hate this product!"
    sentiment_type, subjectivity, positive, neutral, negative, compound_score = get_sentiment(text)
    assert sentiment_type == "Negative"
    assert compound_score <= -0.05
    assert negative > positive and negative > neutral

def test_get_sentiment_neutral():
    text = "The product performs adequately."
    sentiment_type, subjectivity, positive, neutral, negative, compound_score = get_sentiment(text)
    assert sentiment_type == "Neutral"
    assert -0.05 < compound_score < 0.05

# Coded with help from ChatGPT
def test_generate_wordcloud():
    text = "I love Streamlit! Streamlit is awesome!"
    with patch("streamlit.image") as mock_image:
        generate_wordcloud(text)

        mock_image.assert_called_once()
        
        wordcloud_image = mock_image.call_args[0][0]
        assert isinstance(wordcloud_image, np.ndarray)