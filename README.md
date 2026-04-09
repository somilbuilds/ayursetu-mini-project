# Ayur-Setu: Smart Feedback Analyzer

**Ayur-Setu** is an AI-based Patient Feedback Analysis desktop application built specifically for Panchakarma and Ayurvedic centers. It uses Natural Language Processing (NLP) and Machine Learning (Logistic Regression) to automatically classify patient feedback into Positive, Negative, or Neutral sentiments.

## 🚀 Features

- **AI Sentiment Analysis:** Automatically predicts the sentiment of patient feedback using a TF-IDF and Logistic Regression pipeline.
- **Interactive Dashboard:** Beautiful desktop UI built with Tkinter featuring patient feedback tables, detailed AI summaries, and interactive popups.
- **Data Visualization:** Built-in dashboard (powered by Matplotlib) demonstrating Sentiment Distribution, Sentiment Share (Pie chart), and Therapy-wise Sentiment breakdown.
- **Active Learning Feature:** Medical staff can manually correct the AI's label from the UI, and the model will automatically retrain on the new data to improve its accuracy!
- **Data Persistence:** Keeps track of patient records securely using a local CSV file. Loads model state continuously via Python Pickle.

## 🛠️ Technology Stack

- **Frontend UI:** Python Tkinter (`tkinter`, `ttk`)
- **Machine Learning:** Scikit-Learn (`LogisticRegression`, `TfidfVectorizer`, `Pipeline`)
- **Data Manipulation:** Pandas
- **Visualization:** Matplotlib
- **Persistent Storage:** CSV, Pickle

## ⚙️ How to Run

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
3. Run the main script:
   ```bash
   python ayur_setu.py
   ```
4. A Tkinter window will pop up. Click **"Load & Analyze"** to get started!

## 🎓 Academic Mini-Project
This project was developed as a college AI lab mini-project to demonstrate the practical use of NLP models and GUI interactions.
