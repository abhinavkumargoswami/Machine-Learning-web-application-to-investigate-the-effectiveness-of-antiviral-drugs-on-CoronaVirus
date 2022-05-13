import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def convert_df(df):
    return df.to_csv().encode('utf-8')

def load_ml_data():
    df = pd.read_csv("antiviral_preds_without_id.csv")
    X = df.loc[:, df.columns != 'pIC50']
    y = df['pIC50']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def load_pred_data():
    df = pd.read_csv("final_antiviral_props_without_id.csv")
    return df

def load_ids():
    df = pd.read_csv("final_antliviral_ids.csv")
    return df[['CID','IUPACName','MolecularFormula']]


st.title('Machine Learning web application to investigate the effectiveness of antiviral drugs on CoronaVirus')
st.sidebar.subheader("Choose Model")
model_name = st.sidebar.selectbox("Model", ("DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor","LinearRegression"))

if model_name == "LinearRegression":
    st.sidebar.subheader("Hyperparameters")

    if st.sidebar.button("Predict", key="Predict"):
        st.subheader("Linear Regression Results")
        model = LinearRegression()
        X_train, X_test, y_train, y_test = load_ml_data()
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        X_pred = load_pred_data()
        y_pred = model.predict(X_pred).flatten()
        data = load_ids()
        data['Predicted pIC50'] = y_pred
        data = data.sort_values(by='Predicted pIC50', ascending=False)
        st.dataframe(data)
        csv = convert_df(data)
        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )
        st.write("coefficient of determination: ", r2.round(2))


if model_name == "DecisionTreeRegressor":
    st.sidebar.subheader("Hyperparameters")
    Criterion = st.sidebar.selectbox("Criterion", ("mse", "friedman_mse", "mae"), 0)
    splitter = st.sidebar.selectbox("splitter", ("best", "random"), 0)
    max_depth = st.sidebar.slider("max_depth",0,500,0)
    if max_depth == 0:
        max_depth = None
    random_state = int(st.sidebar.number_input("random_state",0))
    if st.sidebar.button("Predict", key="Predict"):
        st.subheader("DecisionTreeRegressor Results")
        model = DecisionTreeRegressor(criterion=Criterion, splitter=splitter,
                                      max_depth=max_depth, random_state=random_state)
        X_train, X_test, y_train, y_test = load_ml_data()
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        X_pred = load_pred_data()
        y_pred = model.predict(X_pred).flatten()
        data = load_ids()
        data['Predicted pIC50'] = y_pred
        data = data.sort_values(by='Predicted pIC50', ascending=False)
        st.write("coefficient of determination: ", r2.round(2))
        st.dataframe(data)
        csv = convert_df(data)
        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

if model_name == "RandomForestRegressor":
    st.sidebar.subheader("Hyperparameters")
    n_estimators = int(st.sidebar.number_input('n_estimators',0,1000,100))
    Criterion = st.sidebar.selectbox("Criterion", ("mse", "mae"), 0)
    max_depth = st.sidebar.slider("max_depth", 0, 500, 0)
    if max_depth == 0:
        max_depth = None
    random_state = int(st.sidebar.number_input("random_state", 0))
    if st.sidebar.button("Predict", key="Predict"):
        st.subheader("RandomForestRegressor Results")
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=Criterion,
                                      max_depth=max_depth, random_state=random_state)
        X_train, X_test, y_train, y_test = load_ml_data()
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        X_pred = load_pred_data()
        y_pred = model.predict(X_pred).flatten()
        data = load_ids()
        data['Predicted pIC50'] = y_pred
        data = data.sort_values(by='Predicted pIC50', ascending=False)
        st.write("coefficient of determination: ", r2.round(2))
        st.dataframe(data)
        csv = convert_df(data)
        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

if model_name == "GradientBoostingRegressor":
    st.sidebar.subheader("Hyperparameters")
    n_estimators = int(st.sidebar.number_input('n_estimators',0,1000,100))
    loss = st.sidebar.selectbox("Loss", ("ls", "lad", "huber", "quantile"), 0)
    learning_rate = st.sidebar.number_input("learning_rate", 0.01)
    Criterion = st.sidebar.selectbox("Criterion", ("friedman_mse", "mse", "mae"), 0)
    max_depth = st.sidebar.slider("max_depth", 1, 500, 3)
    random_state = int(st.sidebar.number_input("random_state", 0))
    if st.sidebar.button("Predict", key="Predict"):
        st.subheader("GradientBoostingRegressor Results")
        model = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                          criterion=Criterion, max_depth=max_depth, random_state=random_state)
        X_train, X_test, y_train, y_test = load_ml_data()
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        X_pred = load_pred_data()
        y_pred = model.predict(X_pred).flatten()
        data = load_ids()
        data['Predicted pIC50'] = y_pred
        data = data.sort_values(by='Predicted pIC50', ascending=False)
        st.write("coefficient of determination: ", r2.round(2))
        st.dataframe(data)
        csv = convert_df(data)
        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )



