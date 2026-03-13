# 🎬 Movie Recommendation System

This project builds a **content-based movie recommendation system** using Natural Language Processing (NLP) and cosine similarity.

The system recommends similar movies based on **genres, keywords, cast, crew, and overview**.

The model is deployed using **Streamlit**, allowing users to select a movie and get recommendations with posters.

---

# 🚀 Features

• Content-based movie recommendation  
• NLP text preprocessing  
• Cosine similarity for recommendation  
• Streamlit interactive UI  
• Movie posters fetched using TMDB API  

---

# 📊 Dataset

The dataset used is the **TMDB 5000 Movies Dataset** containing information about:

- Movie overview
- Genres
- Keywords
- Cast
- Crew
- Movie IDs

Dataset files:
tmdb_5000_movies.csv
tmdb_5000_credits.csv

Source: kaggle

---

# 🧠 Methodology

The recommendation system follows these steps:

### 1 Data Preprocessing
- Merge movies and credits dataset
- Remove null values
- Extract genres, keywords, cast, and director

### 2 Feature Engineering
Combine important movie features into a **tags column**

tags = overview + genres + keywords + cast + crew

### 3 NLP Processing
- Convert text to lowercase
- Apply **stemming using PorterStemmer**

Example:
dancing → danc
dance → danc

### 4 Vectorization
Text features are converted into vectors using:
CountVectorizer(max_features=5000)

### 5 Similarity Calculation
Cosine similarity is used to measure movie similarity.
cosine_similarity(vectors)

---

# 🎥 Example Recommendation

If the user selects:
Batman Begins

The system might recommend:
The Dark Knight
Man of Steel
Superman Returns
Batman
The Dark Knight Rises

---

# 🖥 Streamlit Application

The Streamlit app allows users to:
1️⃣ Select a movie from the dropdown  
2️⃣ Click **Show Recommendation**  
3️⃣ Get **5 recommended movies with posters**

Posters are fetched using the **TMDB API**.

You can run the application using "streamlit run app.py" command 

---

# 👨‍💻 Author
Nakul Jain  
Machine Learning Enthusiast
