import streamlit as st
import pickle
import numpy as np
import base64
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the model using Pickle
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_class(variance, skewness, kurtosis, entropy):
    features = np.array([[variance, skewness, kurtosis, entropy]])
    prediction = model.predict(features)
    return prediction[0]

csv_file_path = "BankNote_Authentication.csv"    
df = pd.read_csv(csv_file_path)

features = ['variance', 'skewness', 'curtosis', 'entropy']
X = df[features]
y = df['class']

# Split the dataset into a 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Home page
def home_page():
    st.title("Welcome to the BankNote Authenticator App 💵")
    st.markdown("""
    **Every country incorporates a set
                 of security features on a banknote 
                for economic protection. 
                However, counterfeiters find multiple ways to forge banknotes and replace genuine banknotes. A counterfeit currency can be defined as an illegal imitation of the national or state currency created to seem as if it is approved by the government. Counterfeit banknotes have always imposed an obstacle in the monetary system worldwide.**
    
    **There is an emerging need for 
                an automated system to identify
                 counterfeit banknotes to be installed at banks. Hence various supervised learning 
                algorithms of machine learning like Logistic Regression, random forest classifier, decision tree, AdaBoost, and support vector machine (SVM) to finalize the most accurate model to distinguish between a forged banknote and a genuine banknote.**         
                """,True)
    
    st.subheader('About Dataset')
    st.markdown("""
    **Data were extracted from images that were taken from genuine and forged banknote-like specimens. 
                For digitization, an industrial camera usually used for print inspection was used. 
                The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.**
                 """,True)

    # Display the original DataFrame
    st.subheader('Original DataFrame:')
    st.dataframe(df)

    # Display descriptive statistics using describe
    st.subheader('Descriptive Statistics:')
    st.dataframe(df.describe())

    st.subheader('Why SVM?')
    st.markdown("""
    **The Final Model chosen through performance metrics is the SVM model. The User is asked to enter the input features such as variance,skewness,kurtosis and entropy.
                Class 0 predicts the banknote is forged while class 1 denotes the banknote is original.**         
                """,True)
    st.image("compare.png", width=400)

# Prediction page
def prediction_page():
    st.title("BankNote Authenticator 🔎")

    # User input for prediction
    st.subheader("Enter parameters for prediction:")
    variance = st.number_input("Variance", value=0.0)
    skewness = st.number_input("Skewness",  value=0.0)
    kurtosis = st.number_input("Kurtosis",  value=0.0)
    entropy = st.number_input("Entropy",  value=0.0)

    # Make prediction
    if st.button("Predict"):
        prediction_result = predict_class(variance, skewness, kurtosis, entropy)
        if prediction_result == 1:
            st.success(f"Prediction: {prediction_result}")
            st.success('The Banknotes are Genuine ✔️')
        else:
            st.error(f"Prediction: {prediction_result}")
            st.error('The Banknotes are Forged ⚔️')

#Metrics page
def metrics_page():
    st.title("Performance Metrics 📊")

    accuracy = accuracy_score(y_test, y_pred)
    st.subheader('Accuracy:')
    st.write(f'The accuracy of the SVM model is: {accuracy:.2%}')

    # Display the confusion matrix as a heatmap
    st.subheader('Confusion Matrix:')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot()

# Streamlit app
def main():
    st.set_page_config(page_title="Banknote Authentication App", layout="wide")
   
    st.markdown(
                f"""
                <style>
                .stApp {{
                    background: url("https://img.freepik.com/premium-photo/falling-banknotes-concept_23-2148542410.jpg?size=626&ext=jpg&ga=GA1.1.716560133.1702479546&semt=ais");
                    background-size: cover
                }}
                
                </style>
                """,
                unsafe_allow_html=True
            )
    st.markdown(
        """<style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 50px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation with selectbox
    selected_page = st.sidebar.radio("Navigate", ["Home", "Prediction","Performance"])
    if selected_page == "Home":
        home_page()

    elif selected_page == "Prediction":
        prediction_page()
    
    elif selected_page == "Performance":
        metrics_page()

if __name__ == "__main__":
    main()
