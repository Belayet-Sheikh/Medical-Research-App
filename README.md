# 🩺 Medical Literature Intelligence

This is an AI-powered web application built with Streamlit and Google Gemini that allows users to analyze and query online medical articles.

![App Screenshot](https://imgur.com/tyIoh1g)

## Features

- **AI-Powered Summarization:** Generates a concise summary of the content from up to three article URLs.
- **Intelligent Q&A:** Allows users to ask specific questions and receive answers sourced directly from the provided articles.
- **Professional UI:** A clean, medical-themed interface for a user-friendly experience.
- **Powered by Google Gemini:** Utilizes the `gemini-1.5-flash` model for fast and accurate text generation.

## How to Deploy and Run

This application is designed to be deployed on Streamlit Community Cloud.

1.  **Fork/Clone this Repository:** Get a copy of this project on your own GitHub account.

2.  **Sign up for Streamlit Community Cloud:** Use your GitHub account to sign up for free at [share.streamlit.io](https://share.streamlit.io/).

3.  **Create API Keys:** You will need a `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/).

4.  **Deploy the App:**

    - In Streamlit Cloud, click "New app".
    - Choose your forked repository and the `main` branch.
    - Ensure the "Main file path" is `app.py`.
    - Click "Advanced settings".

5.  **Add Your Secrets:**

    - In the "Secrets" section, add your Google API key like this:
      ```toml
      GOOGLE_API_KEY = "AIzaSy...YOUR_REAL_KEY_HERE"
      ```
6.  **Click "Deploy!"** and your application will be live.
