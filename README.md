# Auto Email / Ticket Categorizer

This project was created as part of an AI/ML internship assessment.

The goal of the project is to automatically classify customer support tickets into the correct department using Natural Language Processing (NLP) and Machine Learning.

Live link streamlit - https://ticket-categorizer-lwxyp9xhnp2pgcnaez22gd.streamlit.app/
## Categories

The model predicts one of the following categories:

- Billing
- Technical
- HR
- General

## Features

- Text preprocessing
- TF-IDF Vectorization
- Multinomial Naive Bayes classifier
- Model evaluation with accuracy and classification report
- Confidence score for predictions
- Human review suggestion for low-confidence predictions
- Priority detection for urgent tickets
- Simple Streamlit web interface

## Dataset

A dummy dataset was created specifically for this assessment. It contains support tickets from four different categories:

- Billing
- Technical
- HR
- General

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## Project Structure

```
ticket-categorizer/
│── dataset.csv
│── train.py
│── app.py
│── model.pkl
│── vectorizer.pkl
│── confusion_matrix.png
│── requirements.txt
└── README.md
```

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Run the application

```bash
streamlit run app.py
```

## Sample Inputs

**Billing**
```
My payment failed but money was deducted.
```

**Technical**
```
The application crashes after login.
```

**HR**
```
I need my salary slip.
```

**General**
```
What are your office timings?
```

## Model

- TF-IDF Vectorizer
- Multinomial Naive Bayes

## Results

The model is trained on the dummy dataset and predicts the category of a support ticket along with:

- Predicted category
- Confidence score
- Priority level
- Human review suggestion (when confidence is low)

## Future Improvements

- Train on a larger dataset
- Improve text preprocessing
- Try advanced NLP models such as BERT
- Deploy the application online

---

This project was developed for an AI/ML internship assessment to demonstrate the basics of text classification using machine learning.
