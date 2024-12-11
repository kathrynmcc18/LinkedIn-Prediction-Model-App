import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st

#Title
st.title("LinkedIn User Prediction")

#Set custom CSS for the background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0A66C2;  /* Change to your desired color */
    }

    .stTextInput label, .stSelectbox label, .stNumberInput label {
        font-size: 24px;  /* Make the font size bigger */
        color: white;     /* Change the text color to white */
    }

    /* Make the title white */
    .streamlit-expanderHeader {
        color: white;  /* Change the title color to white */
    }

    h1 {
        color: white !important;  /* Ensure h1 elements are white */
    }

    /* Make prediction and probability output white and bigger font */
    .stMarkdown, .stWrite {
        color: white !important; /* White text color */
        font-size: 24px;         /* Bigger font size */
        font-weight: bold;       /* Bold the text */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

#Variable Inputs

age = st.number_input("Enter age:", min_value = 0, max_value = 130)
income = st.selectbox("Income level:", ["Less than $10,000", "10 to under $20,000", "20 to under $30,000", "30 to under $40,000", "40 to under $50,000",
                                         "50 to under $75,000", "75 to under $100,000", "100 to under $150,000", "$150,000 or more"])
education = st.selectbox("Education level:", ["Less than high school (Grades 1-8 or no formal schooling)", "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)", "High school graduate (Grade 12 with diploma or GED certificate)",
                                             "Some college, no degree (includes some community college)", "Two-year associate degree from a college or university", "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                                             "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)", "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])
married = st.selectbox("Marital Status:", ["Single", "Married"])
parent = st.selectbox("Do you have children under 18 living in your home?:", ["Yes", "No"])
female = st.selectbox("Gender:", ["Female", "Male"])

#Convert inputs to numeric variables
income_mapping = {
    "Less than $10,000": 1,
    "10 to under $20,000": 2,
    "20 to under $30,000": 3,
    "30 to under $40,000": 4,
    "40 to under $50,000": 5,
    "50 to under $75,000": 6,
    "75 to under $100,000": 7,
    "100 to under $150,000": 8,
    "$150,000 or more": 9
}
income = income_mapping[income]

education_mapping = {
    "Less than high school (Grades 1-8 or no formal schooling)": 1,
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)": 2,
    "High school graduate (Grade 12 with diploma or GED certificate)": 3,
    "Some college, no degree (includes some community college)": 4,
    "Two-year associate degree from a college or university": 5,
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)": 6,
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)": 7,
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)": 8
}
education = education_mapping[education]

married = 1 if married == "Married" else 0
parent = 1 if parent == "Yes" else 0
female = 1 if female == "Female" else 0


s = pd.read_csv("social_media_usage.csv")

def clean_sm(x): 
    return np.where(x==1,1,0)


toy = {
    "a": [1, 2, 3],
    "b": [13, 14, 15],
    "c": [1,1,1]
}

toy = pd.DataFrame(toy)

toy_cleaned = toy.applymap(clean_sm)

# Clean data
ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]), 
    "parent": clean_sm(s["par"]),
    "married": clean_sm(s["marital"]),
    "female": np.where(s["gender"]==2,1,0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility
# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.

#Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')

# Fit algorithm to training data
lr.fit(X_train, y_train)

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

# Compare those predictions to the actual test data using a confusion matrix (positive class=1)
#confusion_matrix(y_test, y_pred)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

print(classification_report(y_test, y_pred))

# New data for predictions
newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]
})

# Use model to make predictions
newdata["prediction_has_linkedin"] = lr.predict(newdata)


# New data for features: age, income, education, married, parent, female
person = [income, education, parent, married, female, age]

# Add a button to trigger prediction
if st.button('Am I a LinkedIn User?'):
    try:
        # Use model to make prediction
        predicted_class = lr.predict([person])
        
        # Get the probability of the positive class (LinkedIn user)
        probs = lr.predict_proba([person])

        # Convert probability to percentage and round to 2 decimal places
        probability_percent = round(probs[0][1] * 100, 2)
        
        # Display the prediction
        st.write(f"Prediction: {'LinkedIn User' if predicted_class[0] == 1 else 'Not a LinkedIn User'}")
        st.write(f"Probability that this individual uses LinkedIn: {probability_percent}%")
    except Exception as e:
        st.error(f"Error: {e}")
