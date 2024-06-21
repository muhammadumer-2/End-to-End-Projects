#Import Libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

#---------------------------------------------------------------------------------------------------

# IRIS Data Set Prediction Model 
menu = ["Select the Option","Iris Flower Classification","Credit card fraud detection"]
choice = st.sidebar.selectbox("Main_menu",menu)
credit_card = pd.read_csv("creditcard.csv")

def iris_classification_model():
    # Load the IRIS dataset
    df = pd.read_csv("IRIS.csv")
    
    # Split the data into features (X) and target (Y)
    X = df.drop("species", axis=True)
    Y = df["species"]
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.25)
    
    # Create and train an SVM model
    svc = SVC()
    svc_model = svc.fit(X_train, Y_train)
    
    # Evaluate the model
    predic = svc_model.predict(X_test)
    accuracy = accuracy_score(predic, Y_test)
    #st.write("Model Accuracy:", accuracy)
    
    # Create a Streamlit interface to input new data points
    st.write("Enter new data point:")
    sepal_length = st.number_input("Enter sepal length:")
    sepal_width = st.number_input("Enter sepal width:")
    petal_length = st.number_input("Enter petal length:")
    petal_width = st.number_input("Enter petal width:")
    
    # Create a new input data point
    new_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    # Predict the species using the trained model
    if st.button("Predict"):
        predicted_species = svc_model.predict(new_input)
        st.write("Predicted species:", predicted_species[0])

#--------------------------------------------------------------------------------------------

#Credit card fraud Detection Model

def credit_card_fraud_detection_model():
    credit_card = pd.read_csv("creditcard.csv")
    X = credit_card.drop("Class", axis=True)
    Y = credit_card["Class"]
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=42)
    smote = SMOTE(random_state=0)
    x_train, y_train = smote.fit_resample(X_Train, Y_Train)
    x_test, y_test = smote.fit_resample(X_Test, Y_Test)
    rf = RandomForestClassifier(max_depth=9, n_estimators=50)
    RF_clf = rf.fit(x_train, y_train)
    pre = RF_clf.predict(x_test)
    acc = accuracy_score(pre, y_test)
    re = recall_score(pre, y_test)
    f1 = f1_score(pre, y_test)
    return RF_clf

def predict_fraud(RF_clf, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount):
    input_array = np.array([[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]]).astype(np.float64)
    prediction = RF_clf.predict(input_array)
    return prediction

def create_streamlit_app(RF_clf):
    st.title("Credit Card Fraud Detection")
    st.write("Enter the values for each feature:")
    V1 = st.number_input("V1:")
    V2 = st.number_input("V2:")
    V3 = st.number_input("V3:")
    V4 = st.number_input("V4:")
    V5 = st.number_input("V5:")
    V6 = st.number_input("V6:")
    V7 = st.number_input("V7:")
    V8 = st.number_input("V8:")
    V9 = st.number_input("V9:")
    V10 = st.number_input("V10:")
    V11 = st.number_input("V11:")
    V12 = st.number_input("V12:")
    V13 = st.number_input("V13:")
    V14 = st.number_input("V14:")
    V15 = st.number_input("V15:")
    V16 = st.number_input("V16:")
    V17 = st.number_input("V17:")
    V18 = st.number_input("V18:")
    V19 = st.number_input("V19:")
    V20 = st.number_input("V20:")
    V21 = st.number_input("V21:")
    V22 = st.number_input("V22:")
    V23 = st.number_input("V23:")
    V24 = st.number_input("V24:")
    V25 = st.number_input("V25:")
    V26 = st.number_input("V26:")
    V27 = st.number_input("V27:")
    V28 = st.number_input("V28:")
    Amount = st.number_input("Amount:")
    if st.button("Predict"):
        prediction = predict_fraud(RF_clf, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount)
        st.write("Prediction:", prediction)


#Streamlit App code

if choice == "Iris Flower Classification":
    menu = ["Select the Option","About","Data Set","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "About":
        
        st.markdown("<h1 style='text-align: center; color: blue;'>IRIS SPECIES PREDICTION About Dataset</h1>", unsafe_allow_html=True)
        st.write("The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.")
        st.write("This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines")
        st.write("The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).")    
    
    
    elif choice == "Data Set":
        
        st.markdown("<h1 style='text-align: center; color: blue;'>IRIS SPECIES PREDICTION DATA SET...!</h1>", unsafe_allow_html=True)
        df = pd.read_csv("IRIS.csv")
        st.write(df)
        st.download_button(
            label=" Download ",
            data=df.to_csv(index=False),
            file_name="IRIS.csv",
            mime="text/csv"
            )   
        
    elif choice == "Prediction":
        
        st.markdown("<h1 style='text-align: center; color: blue;'>IRIS SPECIES PREDICTION...!</h1>", unsafe_allow_html=True)
        iris_classification_model()
        

elif choice == "Credit card fraud detection":
    menu = ["Select the Option","About","Data Set","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "About":
        st.markdown("<h1 style='text-align: center; color: blue;'>Credit card fraud detection About Dataset</h1>", unsafe_allow_html=True)
        st.write("It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.")
        st.write("The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.")
        st.write("It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.")
        st.write("Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.")
        st.write("Update (03/05/2021) A simulator for transaction data has been released as part of the practical handbook on Machine Learning for Credit Card Fraud Detection - https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html. We invite all practitioners interested in fraud detection datasets to also check out this data simulator, and the methodologies for credit card fraud detection presented in the book.")
        st.write("Acknowledgements The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project")
        

    elif choice == "Data Set":
        
        st.markdown("<h1 style='text-align: center; color: blue;'>Credit card fraud detection Dataset...!</h1>", unsafe_allow_html=True)
        st.write(credit_card)
        st.download_button(
            label=" Download ",
            data=credit_card.to_csv(index=False),
            file_name="Credit_Card.csv",
            mime="text/csv"
            )

    elif choice == "Prediction":
        
        st.markdown("<h1 style='text-align: center; color: blue;'>Credit card fraud detection Model...!</h1>", unsafe_allow_html=True)
        RF_clf = credit_card_fraud_detection_model()
        create_streamlit_app(RF_clf)