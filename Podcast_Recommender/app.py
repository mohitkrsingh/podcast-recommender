import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Podcast Recommender", layout="centered")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("df_popular_podcasts.csv")
    return df

df = load_data()

# Show title
st.title("🎙️ Podcast Recommender System")
st.markdown("Find podcasts similar to your favorite ones based on descriptions.")

# Vectorize descriptions
@st.cache_resource
def get_similarity_matrix(descriptions):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(descriptions)
    return cosine_similarity(tfidf_matrix)

similarity_matrix = get_similarity_matrix(df['cleaned_description'].fillna(""))

# Select a podcast
selected_podcast = st.selectbox("Choose a podcast you like:", df['Name'])

# Recommend podcasts
def recommend_podcasts(podcast_name, top_n=5):
    idx = df[df['Name'] == podcast_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return df.iloc[top_indices][['Name', 'Description', 'iTunes URL', 'Podcast URL']]

if st.button("Recommend"):
    st.subheader("Recommended Podcasts:")
    recommendations = recommend_podcasts(selected_podcast)
    for i, row in recommendations.iterrows():
        st.markdown(f"### 🎧 {row['Name']}")
        st.write(row['Description'][:400] + "...")
        st.markdown(f"[iTunes]({row['iTunes URL']}) | [Website]({row['Podcast URL']})")
        st.markdown("---")


