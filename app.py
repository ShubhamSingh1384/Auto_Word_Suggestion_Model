import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load Saved Assets ---
st.title("🧠 Auto-Word Suggestion Model")
st.write("Enter a phrase and let the model predict the next words!")

@st.cache_resource
def load_all_assets():
    # Load the trained LSTM model
    model = load_model('lstm_model.h5')

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load max_sequence_len
    with open('max_sequence_len.pkl', 'rb') as handle:
        max_seq_len = pickle.load(handle)
        
    # Create index_to_word mapping
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    
    return model, tokenizer, max_seq_len, index_to_word

model, tokenizer, max_sequence_len, index_to_word = load_all_assets()

# --- 2. Define Prediction Functions (re-using the ones from the notebook) ---
def predict_next_word(model, tokenizer, current_text, max_sequence_len, index_to_word):
    token_list = tokenizer.texts_to_sequences([current_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    
    predicted_probabilities = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probabilities)
    
    next_word = index_to_word.get(predicted_index, '<unk>')
    
    return next_word

def predict_next_line(model, tokenizer, seed_text, max_sequence_len, index_to_word, n_words):
    full_text = seed_text
    for _ in range(n_words):
        next_word = predict_next_word(model, tokenizer, full_text, max_sequence_len, index_to_word)
        full_text += " " + next_word
    return full_text

# --- 3. Streamlit UI ---

seed_phrase = st.text_input("Enter your seed phrase:", "life isnt finding")

num_words = st.slider("Number of words to predict for the line:", 1, 10, 5)

if st.button("Predict Next Word"):
    if seed_phrase:
        predicted_single_word = predict_next_word(model, tokenizer, seed_phrase, max_sequence_len, index_to_word)
        st.success(f"**Next word prediction:** `{predicted_single_word}`")
    else:
        st.warning("Please enter a seed phrase.")

if st.button("Predict Next Line (Multiple Words)"):
    if seed_phrase:
        predicted_full_line = predict_next_line(model, tokenizer, seed_phrase, max_sequence_len, index_to_word, num_words)
        st.success(f"**Predicted line:** `{predicted_full_line}`")
    else:
        st.warning("Please enter a seed phrase.")

st.write("---")
st.info("To run this Streamlit app:\n1. Save this code as `app.py` in the same directory as `lstm_model.h5`, `tokenizer.pkl`, and `max_sequence_len.pkl`.\n2. Open your terminal in that directory.\n3. Run `streamlit run app.py`")