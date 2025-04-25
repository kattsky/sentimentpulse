# Code adapted and modified from: https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation
# https://cheat-sheet.streamlit.app/?ref=blog.streamlit.io

# Description: This file contains the navigation setup and runs the app.

import streamlit as st

# --- PAGE SETUP ---

main_app = st.Page(
    page="views/app.py",
    title="SentimentPulse",
    icon=":material/psychology:",
    default=True,
)
about_page = st.Page(
    page="views/2_about.py",
    icon= ":material/settings:",
    title="About",
)
contact_page = st.Page(
    page="views/3_contact.py",
    title="Contact",
    icon=":material/contact_page:",
)

# --- NAVIGATION SETUP ---
nav = st.navigation(pages=[main_app, about_page, contact_page])
st.set_page_config(
    page_title="SentimentPulse", page_icon=":material/search_insights:",
    layout="wide",
    initial_sidebar_state="expanded",    
    )

# --- RUN NAVIGATION ---
nav.run()