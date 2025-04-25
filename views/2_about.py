# Description: This page outlines the SentimentPulse app, including its purpose, features, tech stack, attributions, and how it analyzes sentiment in text data.

import streamlit as st

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
        <p>Transform Text Data into Actionable Insights</p>
    </div>

    <div class="purpose">
        <h2 class="header">What is this app for?</h2>
        <p>SentimentPulse leverages advanced sentiment analysis to process text data and extract meaningful insights. It helps users understand public perception, sentiment trends, and emotions from textual content like reviews, feedback, or social media posts.
    </p>
    </div>

    <div class="target-audience">
        <h2 class="header">Who is this app for?</h2>
        <p>Users need a simple, intuitive app that allows them to quickly paste text and receive accurate sentiment analysis. This enables them to make timely, informed decisions for business, personal use, or research.</p>
        <p>The following are the <strong>target audiences</strong>:</p>
        <div>
            <ul>
                <li>Social Media Influencers</li>
                <li>Bloggers and Writers</li>
            </ul>
            <ul>
                <li>Marketing Teams</li>
                <li>Market Researchers</li>
            </ul>
            <ul>
                <li>Customer Service Teams</li>
                <li>Small to Medium Enterprises</li>
                <li>General Consumers</li>
            </ul>
        </div>
    </div>
"""
html_content = html_main
st.html(html_content)

div = """<div class = 'test1'>"""
divEnd = """</div>"""

with st.container() as mvp_features:
    mvp, _, features = st.columns([1.5, .2, 1.5])

    with mvp:
        st.markdown(div, unsafe_allow_html=True)
        st.html("""
        <h3>Minimum Viable Product</h3>
        <ul>
            <li>Text input – allow users to enter text into the text field</li>
            <li>Text preprocessing</li>
            <li>Sentiment classification (positive, neutral, negative)</li>
            <li>Basic data visualization with data frame</li>
            <li>Generate simple feedback output</li>
        </ul>
        """)
        st.markdown(divEnd, unsafe_allow_html=True)

    with features:
        st.html("""
        <h3>Features</h3>
        <ul>
            <li>File Upload with CSV and Excel</li>
            <li>Custom color mapping on dataframe and pie chart.</li>
            <li>Show progress indicator while processing data.</li>
            <li>Generate a word cloud on most frequent words.</li>
            <li>Download processed data as a CSV or Excel file.</li>
        </ul>
        """)
    
with st.container() as tech_stack:
    st.html('<h2 class="header-centered tech-stack">Technologies Used</h2>')

    frontend, backend, ml = st.columns(3)

    with frontend:
        st.html("""
        <h3>Frontend</h3>
        <ul>
            <li>Streamlit</li>
            <li>HTML/CSS</li>
        </ul>
        """)

    with backend:
        st.html("""
        <h3>Backend</h3>
        <ul>
            <li>Python</li>
            <li>Pandas</li>
        </ul>
        """)

    with ml:
        st.html("""
        <h3>Machine Learning</h3>
        <ul>
            <li>NLTK</li>
            <li>TextBlob</li>
            <li>Vader</li>
        </ul>
        """)

with st.container() as attributions:
    st.html('<h2 class="header-centered attribution">Attributions</h2>')
    
    with st.expander("References", expanded=False):
        st.html("""
            <ul>
                <li><a href="https://docs.streamlit.io/">Streamlit documentation</a></li>
                <li><a href="https://discuss.streamlit.io/">Streamlit Discuss</a></li>
                <li><a href="https://docs.python.org/3/">Python documentation</a></li>
                <li><a href="https://www.geeksforgeeks.org/python-programming-language-tutorial/">Python Tutorial</a></li>
                <li><a href="https://stackoverflow.com/questions/tagged/python">Stack Overflow</a></li>

            </ul>
        
        """)
         
    with st.expander("Acknowledgment", expanded=False):
        st.html("""
            <ul>
                <li>Megan Wilson</li>
                <li>Alex Dunae</li>
                <li>Sam Dobson's <a href="https://samdobson-streamlit-sandbox-app-za85j0.streamlit.app/">Streamlit Sandbox</a></li>
            </ul>
        
        """)


   
