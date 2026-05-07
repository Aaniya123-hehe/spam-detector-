import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. SIMPLE DATASET (Spam vs Ham)
# In a real project, you'd load a big CSV file here.
data = {
    'text': [
        'Get a free iPhone now!', 'Call this number for a prize', 
        'Meeting at 10am tomorrow', 'Can you send me the report?',
        'Congratulations! You won a lottery', 'Hey, are we still on for lunch?'
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# 2. THE AI BRAIN (Training)
# Convert text to numbers so the computer can understand
cv = CountVectorizer()
X = cv.fit_transform(df['text'])
model = MultinomialNB()
model.fit(X, df['label'])

# 3. THE WEBSITE UI (What the user sees)
st.title("📧 AI Spam Detector")
st.write("Created by a BBA Student (AI Course Project)")

user_input = st.text_area("Paste the email content below:")

if st.button("Check if Spam"):
    if user_input:
        # Predict
        data_input = cv.transform([user_input]).toarray()
        prediction = model.predict(data_input)
        
        # Display Result
        if prediction[0] == 'spam':
            st.error("🚨 This looks like SPAM!")
        else:
            st.success("✅ This seems like a SAFE email.")
    else:
        st.warning("Please enter some text first.")
