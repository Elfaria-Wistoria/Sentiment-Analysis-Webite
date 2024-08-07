# YouTube Comment Sentiment Analysis

This is a web application for analyzing the sentiment of YouTube comments using a machine learning model. The application allows users to scrape YouTube comments, process and train a sentiment analysis model, classify the sentiment of comments, and test custom comments for sentiment prediction.

## Features

- **Scrape YouTube Comments**: Extract comments from YouTube videos based on a search query.
- **Process Comments**: Clean and prepare the comments for sentiment analysis.
- **Train Model**: Build and train a Support Vector Machine (SVM) model to classify sentiment as positive, negative, or neutral.
- **Classify Comments**: Analyze the sentiment of the processed comments.
- **Test Sentiment**: Allow users to input a custom comment and predict its sentiment using the trained model.

## Technologies Used

- **Flask**: A micro web framework for Python.
- **Flask-Session**: For server-side session management.
- **Tailwind CSS**: A utility-first CSS framework for styling.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Transformers**: For pre-processing text and generating embeddings.
- **Google API Client**: For interacting with the YouTube API.
