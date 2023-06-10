import streamlit as st
import pickle
import numpy as np

# Load the sentiment analysis model
model = pickle.load(open('sentiment_model.pkl', 'rb'))

# Function to perform sentiment analysis
def sentiment_analysis(text):
    # Preprocess the input text (e.g., tokenization, normalization, etc.)
    processed_text = preprocess(text)
    
    # Vectorize the processed text (e.g., using CountVectorizer or TF-IDFVectorizer)
    vectorized_text = vectorize(processed_text)
    
    # Predict the sentiment using the loaded model
    prediction = model.predict(vectorized_text)
    
    # Get the probabilities for each sentiment class
    probabilities = model.predict_proba(vectorized_text)
    
    return prediction, probabilities

# Streamlit app
def main():
    st.title("Sentiment Analysis")
    
    # Input text
    input_text = st.text_input("Enter a sentence:")
    
    # Test button
    if st.button("Test"):
        if input_text:
            # Perform sentiment analysis
            prediction, probabilities = sentiment_analysis(input_text)
            
            # Display the predicted sentiment
            st.write("Sentiment:", prediction)
            
            # Display the probabilities for each sentiment class
            st.write("Probabilities:")
            for sentiment, prob in zip(model.classes_, probabilities.squeeze()):
                st.write(sentiment, ":", np.round(prob, 4))
        else:
            st.warning("Please enter a sentence.")

if __name__ == '__main__':
    main()
