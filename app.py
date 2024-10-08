import pandas as pd
import re
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
import streamlit as st
import string
import tensorflow as tf
import numpy as np

q_words = []
with open('q_words.txt', 'r') as f:
    for i in f:
        i = i.replace('\n', '')
        if i != '':
            q_words.append(i)

# En förenklad textbearbetningsfunktion för att rensa och lemmatisera texten
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

def clean_and_lemmatize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Ta bort skiljetecken
    text = re.sub(r"[^a-z\s']", '', text)  # Endast bokstäver
    text = ' '.join(text.split())  # Ta bort extra mellanslag
    words = text.split()
    # lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(word)
    return lemmatized_words

# Funktion som matchar nyckelord mot en rad i DataFrame
def match_qwords(description, q_words):
    # Hämta texten från 'description' och bearbeta den
    text = clean_and_lemmatize(description)
    # Skapa en lista för att lagra 1:or och 0:or för varje q_word
    word_matches = []
    for word in q_words:
        # Sätt 1 om ordet finns i den bearbetade texten, annars 0
        word_matches.append(1 if word in text else 0)
    return word_matches

def pick_reviews():
    wine = pd.read_csv('winemag-data_first150k.csv', index_col=0)
    wine['country'] = wine['country'].astype('category')
    wine['description'] = wine['description'].astype('string')
    wine['variety'] = wine['variety'].astype('category')
    wine['winery'] = wine['winery'].astype('category')
    wine = wine.drop(columns=["province", "region_1", "region_2", "designation"])
    wine = wine.dropna(subset=["country"])
    wine = wine.drop_duplicates(keep='first')
    wine = wine.dropna(subset=['price'])
    wine.reset_index(drop=True, inplace=True)
    reviews = wine[['variety', 'description']]
    return reviews

df_reviews = pick_reviews()

# Textinmatningsfält
user_input = st.text_area(f"*Add Review:*")

if st.button("Analyze"):
    if user_input == '':
        st.title('Add a review')
    else:
        word_matches = match_qwords(user_input, q_words)
        match_df = pd.DataFrame([word_matches], columns=q_words)
        X = match_df[q_words]
        input_dim = X.shape[1]
            # Återskapa modellen med samma arkitektur
        model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(2, activation='sigmoid'),  
        tf.keras.layers.Dense(2, activation='sigmoid')  
            ])
            # Kompilera modellen innan du laddar vikterna
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # Ladda vikterna
        model.load_weights('colour.weights.h5')
        # st.title("Prediction")
        cleaned_words = clean_and_lemmatize(user_input)
        # st.write(f"**Rensade ord:** {set(cleaned_words)}")
        st.write(f"**Number of matched words:** {sum(word_matches)}")
            # Gör prediktioner
        y_pred_prob = model.predict(X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        var = ['Red', 'White']
        st.title(f"*This Wine Is Most Likely {var[y_pred[0]]}*")
    








