# Description: This page provides contact information for the SentimentPulse app developer.

import streamlit as st
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load external css file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom CSS from style.css
local_css('style.css')

html_main = """
    <div class="main-section">
        <h1>Let's Talk</h1>
        <p>Got questions or feedback about the app? Feel free to reach out!</p>
    </div>
"""
html_content = html_main
st.html(html_content)

# Sending emails with SMTP and Gmail address
# Code adapted from: 
#   https://discuss.streamlit.io/t/send-email-with-smtp-and-gmail-address/48145/2
#   https://realpython.com/python-send-email/#sending-a-plain-text-email

STMTP_SERVER = "smtp.gmail.com"
PORT = 587 # For starttls

# Access Gmail account credentials securely using Streamlit secrets
contact_support = st.secrets["gmail"]["contact_support"]
password = st.secrets["gmail"]["password"]

subject = "SentimentPulse Contact Form Submission"

# Create the email content
def send_email(name, email, message):
    msg = MIMEMultipart()
    msg['Name'] = name
    msg['From'] = email
    msg['Subject'] = subject
    msg['Message'] = message
    body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
    msg.attach(MIMEText(body, 'plain'))

# Connect to the Gmail SMTP server, authenticate, and send email
    try:
        with smtplib.SMTP(STMTP_SERVER, PORT) as server:
            server.starttls()
            server.login(contact_support, password)
            server.sendmail(email, contact_support, msg.as_string())
            server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Function to validate email format
# Code adapted from: https://github.com/streamlit/streamlit/issues/8790
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return None if re.match(pattern, email) else "Please enter a valid email address"

# Contact form code adapted from: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form
with st.form(key='contact_form'):
    contact_info, message = st.columns(2)

    with contact_info:
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
    with message:
        message = st.text_area("Your Message")
    
    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not name or not email or not message:
            st.error("Please fill in all fields.")
        elif validate_email(email):
            st.error(validate_email(email))
        else:
            # Send the email
            if send_email(name, email, message):
                st.success("Form submitted successfully! We will get back to you soon.")
            else:
                st.error("There was an error sending the email. Please try again later.")



st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)

with st.container() as socials:
    st.html("""
        
        <div class="social-icons">
            <h2 class="connect">Let's Connect</h2>
            <div>
                <a href="https://www.linkedin.com/in/katycpl/" target="_blank">
                    <i class="fab fa-linkedin fa-3x"></i>
                </a>
                <a href="https://github.com/kattsky" target="_blank">
                    <i class="fab fa-github-square fa-3x"></i>
                </a>
            </div>
        </div>
    """)