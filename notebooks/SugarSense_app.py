#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all the required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


# loading the dataset
def load_data():
    return pd.read_csv("Diabetes_Dataset.csv")

df = load_data()


# In[3]:


# preprocessing the data
df = df.drop_duplicates()
for col in df.columns:
    df[col] = df[col].astype(int)


# In[4]:


# dropping unnecssary columns
columns_to_drop = ['Income', 'Education', 'NoDocbcCost', 'AnyHealthcare']
df = df.drop(columns=columns_to_drop, inplace=False)


# In[5]:


# defining function to convert age to age range
def age_to_range(age):
    if age >= 18 and age <= 24:
        return 1
    elif age >= 25 and age <= 29:
        return 2
    elif age >= 30 and age <= 34:
        return 3
    elif age >= 35 and age <= 39:
        return 4
    elif age >= 40 and age <= 44:
        return 5
    elif age >= 45 and age <= 49:
        return 6
    elif age >= 50 and age <= 54:
        return 7
    elif age >= 55 and age <= 59:
        return 8
    elif age >= 60 and age <= 64:
        return 9
    elif age >= 65 and age <= 69:
        return 10
    elif age >= 70 and age <= 74:
        return 11
    elif age >= 75 and age <= 79:
        return 12
    else:
        return 13  # 80 or older


# In[6]:


# determining feature selection and model training
selected_features = ['BMI', 'GenHlth', 'HighBP', 'HighChol', 'Age', 'DiffWalk', 'HeartDiseaseorAttack']
X = df[selected_features]
y = df['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model_subset = LogisticRegression()
logistic_model_subset.fit(X_train, y_train)


# In[7]:


# defining function to predict diabetes status
def predict_diabetes_status(bmi, gen_health, high_bp, high_chol, age_range, diff_walk, heart_disease_or_attack):
    user_data = [[bmi, gen_health, high_bp, high_chol, age_range, diff_walk, heart_disease_or_attack]]
    predicted_diabetes_status_subset = logistic_model_subset.predict(user_data)
    return predicted_diabetes_status_subset


# In[8]:


# defining function to plot count of diabetic individuals by BMI category
def plot_bmi_count(bmi_counts, user_input_bmi):
    fig = px.bar(x=bmi_counts.index, y=bmi_counts.values, labels={"x": "BMI Category", "y": "Count"},
                 title="Count of Individuals with Diabetes by BMI Category",
                 color_discrete_sequence=["skyblue"])
    fig.add_vline(x=user_input_bmi, line_dash="dash", line_color="red", annotation_text="Your BMI",
                  annotation_position="top left")
    return fig

# defining function to plot count of diabetic individuals by General Health category
def plot_general_health_count(gen_health_counts, user_input_gen_health):
    fig = px.bar(x=gen_health_counts.index, y=gen_health_counts.values, labels={"x": "General Health Category", "y": "Count"},
                 title="Count of Individuals with Diabetes by General Health Category",
                 color_discrete_sequence=["skyblue"])
    fig.add_vline(x=user_input_gen_health, line_dash="dash", line_color="red", annotation_text="Your General Health",
                  annotation_position="top left")
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5], ticktext=['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']))
    return fig

# defining function to plot count of diabetic individuals by Age category
def plot_age_count(age_counts, user_input_age_range):
    age_labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    fig = px.bar(x=age_counts.index, y=age_counts.values, labels={"x": "Age Category", "y": "Count"},
                 title="Count of Individuals with Diabetes by Age Category",
                 color_discrete_sequence=["skyblue"])
    fig.add_vline(x=user_input_age_range, line_dash="dash", line_color="red", annotation_text="Your Age",
                  annotation_position="top left")
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 14)), ticktext=age_labels))
    return fig

# defining function to plot count of diabetic individuals by High Blood Pressure category
def plot_high_bp_count(high_bp_counts, user_input_high_bp):
    fig = px.bar(x=high_bp_counts.index, y=high_bp_counts.values, labels={"x": "High Blood Pressure Category", "y": "Count"},
                 title="Count of Individuals with Diabetes by High Blood Pressure Category",
                 color_discrete_sequence=["skyblue"])
    fig.add_vline(x=user_input_high_bp, line_dash="dash", line_color="red", annotation_text="Your High Blood Pressure",
                  annotation_position="top left")
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']))
    return fig

# defining function to plot count of diabetic individuals by High Cholesterol category
def plot_high_chol_count(high_chol_counts, user_input_high_chol):
    fig = px.bar(x=high_chol_counts.index, y=high_chol_counts.values, labels={"x": "High Cholesterol Category", "y": "Count"},
                 title="Count of Individuals with Diabetes by High Cholesterol Category",
                 color_discrete_sequence=["skyblue"])
    fig.add_vline(x=user_input_high_chol, line_dash="dash", line_color="red", annotation_text="Your High Cholesterol",
                  annotation_position="top left")
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']))
    return fig


# In[9]:


# defining main function for Streamlit app

def main():
    
    # creating sidebar for user input
    st.sidebar.title("Enter Your Information")
    bmi_known = st.sidebar.radio("Do you know your BMI?", ("Yes", "No"))
    if bmi_known == "Yes":
        bmi = st.sidebar.number_input("Enter BMI", min_value=5, max_value=100)
    else:
        weight_lb = st.sidebar.number_input("Enter weight in pounds")
        height_inches = st.sidebar.number_input("Enter height in inches")
        bmi = 703 * (weight_lb / (height_inches ** 2))
    gen_health_labels = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
    gen_health = st.sidebar.selectbox("General Health", options=list(gen_health_labels.values()))
    gen_health = list(gen_health_labels.keys())[list(gen_health_labels.values()).index(gen_health)]
    high_bp = st.sidebar.radio("Do you have high blood pressure?", ("Yes", "No"))
    high_chol = st.sidebar.radio("Do you have high cholesterol?", ("Yes", "No"))
    age = st.sidebar.number_input("Enter Age", min_value=18, max_value=100)
    age_range = age_to_range(age)
    
    # mapping radio inputs to 0 or 1
    high_bp = 1 if high_bp == "Yes" else 0
    high_chol = 1 if high_chol == "Yes" else 0
    diff_walk = st.sidebar.radio("Do you have difficulty walking?", ("Yes", "No"))
    diff_walk = 1 if diff_walk == "Yes" else 0
    heart_disease_or_attack = st.sidebar.radio("Do you have heart disease or attack?", ("Yes", "No"))
    heart_disease_or_attack = 1 if heart_disease_or_attack == "Yes" else 0

    # determining main content
    st.title("Sugar Sense")
    st.subheader("Predicting Diabetes Risk")
    st.write("Based on the provided information...")

    # predicting live update without submit button
    st.write("\n\n")
    with st.spinner("Predicting..."):
        predicted_diabetes_status = predict_diabetes_status(bmi, gen_health, high_bp, high_chol, age_range, diff_walk, heart_disease_or_attack)
        if predicted_diabetes_status == 1:
            st.error("I'm sorry, but you might be at risk for diabetes. 👎")
        else:
            st.success("Good news! You're not at risk for diabetes. 👍")
            # st.balloons()  # adding balloons animation for good news

    # evaluating your specifics against the data
    if st.checkbox("Evaluate your specifics against the data"):
        option = st.radio("Select a category to evaluate:", ["BMI", "General Health", "High BP", "High Cholesterol", "Age"])
        if option == "BMI":
            bmi_counts = df['BMI'].value_counts().sort_index()
            fig = plot_bmi_count(bmi_counts, bmi)
            st.plotly_chart(fig)
        elif option == "General Health":
            gen_health_counts = df['GenHlth'].value_counts().sort_index()
            fig = plot_general_health_count(gen_health_counts, gen_health)
            st.plotly_chart(fig)
        elif option == "Age":
            age_counts = df['Age'].value_counts().sort_index()
            fig = plot_age_count(age_counts, age_range)
            st.plotly_chart(fig)
        elif option == "High BP":
            high_bp_counts = df['HighBP'].value_counts().sort_index()
            fig = plot_high_bp_count(high_bp_counts, high_bp)
            st.plotly_chart(fig)
        elif option == "High Cholesterol":
            high_chol_counts = df['HighChol'].value_counts().sort_index()
            fig = plot_high_chol_count(high_chol_counts, high_chol)
            st.plotly_chart(fig)

    if st.button('Curious about what Contribution Analysis entails?'):
        st.write("Please take a look of the relative importance of each feature in shaping the result.")
        st.image('features.png', caption='Feature contribution towards the result', width = 500)


# In[10]:


if __name__ == "__main__":
    main()

