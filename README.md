
# Sarcastic_GuRU: A Sarcasm Detection System with Streamlit Interface

This project presents a sarcasm detection system using Natural Language Processing (NLP) and a Bidirectional GRU-based deep learning model. It includes a complete training pipeline, hyperparameter tuning using Keras Tuner, evaluation metrics, and a Streamlit application for interactive predictions.

## Project Overview

- Dataset: Sarcasm Headlines Dataset v2
- Model: Bidirectional GRU (Gated Recurrent Unit)
- Tuning: Keras Tuner (RandomSearch)
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Interface: Streamlit Web Application



## How It Works

1. **Data Preprocessing**
   - Headlines are cleaned by removing special characters, punctuation, and stopwords.
   - Lemmatization is applied to reduce words to their base form.
   - Tokenization and padding are performed for model input.

2. **Model Architecture**
   - Embedding layer to convert words into dense vectors
   - Bidirectional GRU layer with L2 regularization and dropout to avoid overfitting
   - Dense layers for final classification

3. **Model Tuning and Training**
   - Keras Tuner is used to find the optimal hyperparameters.
   - Model is trained with early stopping based on validation loss.

4. **Evaluation**
   - Metrics used include Accuracy, Precision, Recall, and F1-score.
   - Confusion matrix and graphs for loss and accuracy are plotted.

5. **Deployment**
   - The trained model and tokenizer are saved.
   - A Streamlit app allows users to input a headline and receive sarcasm predictions.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/satyy7/sarcastic_GuRU.git
cd sarcastic_GuRU
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Application

The `app.py` file provides an interface where users can enter a news headline. The app loads the saved model and tokenizer, preprocesses the input, and displays whether the statement is sarcastic or not.

## License

This project is provided for educational and non-commercial use.
