# Advanced Capstone Project 2: SentimentPulse

A web app that allows users to analyze the sentiment of any text input, like product reviews, social media comments, and customer service feedback. The tool uses Natural Language Processing techniques to determine the emotional tone of text, providing valuable metrics on sentiment polarity, subjectivity, and compound sentiment scores which helps classifies the text as positive, negative, or neutral.

## 👥 Who is this app for

Users need a simple, intuitive app that allows them to quickly paste text and receive accurate sentiment analysis. This enables them to make timely, informed decisions for business, personal use, or research.

The following are the target audiences:

- Social Media Influencers
- Bloggers and Writers
- Marketing Teams
- Market Researchers
- Customer Service Teams
- Small to Medium Enterprises
- General Consumers

## 🚀 Core Functionality

- Text preprocessing pipeline to clean and normalize raw text with contraction expansion, spell checking, and tokenization.
- Multi-dimensional sentiment analysis that combines two analytical approaches of VADER and TextBlob, with VADER calculating the positive, negative, neutral, and compound sentiment scores and TextBlob determining the subjectivity score to measure opinion vs. fact-based content.
- Visual analytics with styled dataframe highlighting sentiment metrics

## 🌟 Features

- Batch Processing to handle CSV and Excel file upload containing multiple text entries
- Interactive UI with progress indicators during batch processing
- Download sentiment analysis results data
- Multiple data visualization options available:

  1. Styled DataFrames: Color-coded tables highlighting sentiment metrics
  2. Word Clouds: Visual representations of frequent terms with stopword filtering
  3. Pie Charts: Interactive charts showing the distribution of sentiment types

## 🧑‍💻 User Experience Flow

1. Users can input text directly or upload CSV/Excel files containing text data
2. The system processes the input through the preprocessing pipeline
3. Sentiment analysis is performed, generating comprehensive metrics
4. Results are displayed in styled tables and visualizations
5. For batch processing, a progress bar tracks completion status

## 🖥️ Tech Stack

| **Frontend** | **Backend** | **Machine Learning** |
| ------------ | ----------- | -------------------- |
| Streamlit    | Python      | NLTK                 |
| HTML         | Pandas      | TextBlob             |
| CSS          |             | VADER                |

## 💡 Dependencies

streamlit  
textblob  
nltk  
contractions  
pyspellchecker  
vaderSentiment  
pandas  
matplotlib  
openpyxl  
plotly  
wordcloud

## 💡 Attributions

- [Streamlit documentation](https://docs.streamlit.io/)

- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/?ref=blog.streamlit.io)

- [Streamlit Discuss](https://discuss.streamlit.io/)

- [Python documentation](https://docs.python.org/3/)

- [GeeksforGeeks Python Programming Language Tutorial](https://www.geeksforgeeks.org/python-programming-language-tutorial/)

- [Stack Overflow Python Questions](https://stackoverflow.com/questions/tagged/python)

- [Material Icon Library](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded)

- [Inject CSS in Streamlit](https://medium.com/pythoneers/how-to-customize-css-in-streamlit-a-step-by-step-guide-761375318e05)

- [MultiApp Page Tutorial](https://www.youtube.com/watch?v=9n4Ch2Dgex0)

- [How to Control the Layout in Streamlit in 20 Minutes!](https://www.youtube.com/watch?v=saOv9z6Fk88)
