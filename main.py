# import pandas as pd
# import re
# from pathlib import Path
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import tkinter as tk
# import numpy as np
#
# # Load only the specified datasets
# files = [
#     "data/custom_antibiotic_training.csv",
#     "data/expanded_antibiotic_training.csv",
#     "data/large_antibiotic_training.csv"
# ]
#
# dataframes = [pd.read_csv(file) for file in files]
# df = pd.concat(dataframes, ignore_index=True)
#
# # Clean and preprocess data
# df = df.dropna(subset=["Input", "Antibiotic"])
#
# # Encode antibiotic labels
# y = df["Antibiotic"]
# mlb = MultiLabelBinarizer()
# y_vec = mlb.fit_transform([[label] for label in y])
#
# # Vectorize input phrases using basic count vectorizer (no NLP yet)
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# X_vec = vectorizer.fit_transform(df["Input"])
#
# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.2, random_state=42)
#
# # Train the model
# clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
# clf.fit(X_train, y_train)
#
# # Store components for prediction
# trained_model = {
#     "vectorizer": vectorizer,
#     "mlb": mlb,
#     "model": clf
# }
#
# # Prediction function
# def recommend_antibiotics(infection_query, top_n=10):
#     query_vec = trained_model["vectorizer"].transform([infection_query])
#     prediction = trained_model["model"].predict_proba(query_vec)[0]
#     top_indices = prediction.argsort()[::-1][:top_n]
#     top_scores = prediction[top_indices]
#     top_antibiotics = trained_model["mlb"].classes_[top_indices]
#     return pd.DataFrame({"Antibiotic": top_antibiotics, "Confidence": top_scores})
#
# # Example usage
# if __name__ == "__main__":
#     #user_input = input("Enter an infection or disease: ")
#     #print("\nTop antibiotic recommendations:\n")
#
#     root = tk.Tk()
#     root.title('Antibiotic Resistance')
#     root.geometry('400x500')
#     def submit():
#         disease = entry.get()
#         results = recommend_antibiotics(str(entry.get()))
#         text.delete('1.0', tk.END)
#         text.insert(tk.END, results.to_string(index=False))
#
#     entry_title = tk.Label(root,text='Enter the infection or disease: ')
#     entry_title.pack()
#     entry = tk.Entry(root)
#     entry.pack()
#     submit_button = tk.Button(root,text='Submit',command=submit)
#     submit_button.pack()
#
#
#     recommendations = tk.Label(root,text='Top Antibiotic Recommendations')
#     recommendations.pack()
#     text = tk.Text(root,width=40,height=40,bg='sky blue')
#     text.pack()
#     root.mainloop()
# save as app.py
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Load your datasets
files = [
    "data/custom_antibiotic_training.csv",
    "data/expanded_antibiotic_training.csv",
    "data/large_antibiotic_training.csv"
]
dataframes = [pd.read_csv(file) for file in files]
df = pd.concat(dataframes, ignore_index=True)
df = df.dropna(subset=["Input", "Antibiotic"])

# Encode antibiotics
y = df["Antibiotic"]
mlb = MultiLabelBinarizer()
y_vec = mlb.fit_transform([[label] for label in y])

# Vectorize input phrases
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(df["Input"])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.2, random_state=42)
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)

# Function to predict
def recommend_antibiotics(infection_query, top_n=10):
    query_vec = vectorizer.transform([infection_query])
    prediction = clf.predict_proba(query_vec)[0]
    top_indices = prediction.argsort()[::-1][:top_n]
    top_scores = prediction[top_indices]
    top_antibiotics = mlb.classes_[top_indices]
    return pd.DataFrame({"Antibiotic": top_antibiotics, "Confidence": top_scores})

# Streamlit UI
st.title("Antibiotic Recommender")
st.write("Enter an infection or disease to get top antibiotic recommendations.")

user_input = st.text_input("Enter infection or disease:", "")

if user_input:
    results = recommend_antibiotics(user_input)
    st.subheader("Top Antibiotic Recommendations")
    st.dataframe(results)
