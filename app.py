import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import base64

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

data = pd.read_csv('./Data/Salary_Data.csv')

st.markdown("<h1 style='text-align: center;'>Salary Prediction App</h1>", unsafe_allow_html=True)

nav = st.sidebar.radio("Navigation",["Home", "Data", "Salary Prediction", "Contribute"])
st.sidebar.info("Business vector created by [pch.vector](https://www.freepik.com/vectors/business)")
st.sidebar.info("Self Exploratory Visualization on Salary Dataset - Brought to you By [Parth Desai](https://github.com/pycoder2000)")
st.sidebar.text("Built with ❤️ Streamlit")
      
if nav == "Home":
    st.image("./images/8432.jpg", use_column_width=True)
    description = """
        This Simple Linear Regression is a fictional case study based on a [Kaggle dataset](https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression)

        ### Dependencies:
        
        * [Numpy](https://numpy.org/)
        * [Pandas](https://pandas.pydata.org/)
        * [SciKit-Learn](https://scikit-learn.org/stable/)
        * [StatsModels](https://www.statsmodels.org/stable/index.html)
        * [Matplotlib](https://matplotlib.org/)
        * [Seaborn](https://seaborn.pydata.org/)
        

        ### Fictional Case: Company X

        Company X provided you with employee data containing information about its employees experience time and salaries. The company is about to hire a new employee and wants you to estimate the salary for the new employee considering the time of experience.

        ### Linear Regression Model: Overview

        Technique used to describe the relationship between two variables where one variable (the dependent variable denoted by y) is expected to change as the other one (independent,  predictor variable denoted by x) changes. Regression analysis is commonly used for modeling the relationship between a single dependent variable Y and one or more predictors.  When we have one predictor, we call this "simple" linear regression.

        Linear regression is the statistical technique of fitting a straight line to data, where the regression line is: 
        * Y = A + Bx + ε 
        
        Where Y is the predictor variable, A the intercept, B the slope and X the explanatory variable. That is, in a linear regression, Y varies as a function of BX plus a constant value A. When X = 0, A = Y (intercept). The slope B represents the rate at which Y increases in relation to X. From a set of data it is possible to estimate the values of A and B, as well as the errors associated with these parameters. In doing so, we can plot the line that best fits our data, that is, the line that minimizes the sum of squares of the Error.  The value y is the predicted value and the difference between y and the observed value is the error.

        In short:

        * y = dependent variable
        * A = constant (y intercept) 
        * B = gradient (slope coefficient)
        * error = difference between y and the observed value 

        #### Correlation coefficient

        The correlational coefficient is the statistical technique used to measure strength of linear association, r, between two continuous variables, i.e. closeness with which points lie along the regression line, and lies between -1 and +1

        * if r = 1 or -1 it is a perfect linear relationship
        * if r = 0 there is no linear relationship between x & y

        Conventionally:

        * |r| > 0.8 => very strong relationship

        * 0.6 ≤ |r| strong relationship

        * 0.4 ≤ |r| moderate relationship

        * 0.2 ≤ |r| weak relationship

        * |r| very weak relationship

        Note, however, that the statistical significance depends on the sample size. You can test whether r is statistically significantly different from zero. Note that the larger the sample, the smaller the value of r that becomes significant. For example with n=10 pairs, r is significant if it is greater than 0.63. With n=100 pairs, r is significant if it is greater than 0.20.

        Important points:

        * For large samples very weak relationships can be detected
        * The linear correlation coefficient measures the strength and direction of the linear relationship between two variables  x  and  y .
        * The sign of the linear correlation coefficient indicates the direction of the linear relationship between  x  and  y .
        * When  r  is near  1  or  −1  the linear relationship is strong; when it is near  0  the linear relationship is weak.
    """

    st.markdown(description, unsafe_allow_html=True)

if nav == "Data":
    if st.checkbox("Show Table"):
        st.table(data)

    if st.checkbox("Show Graph"):
        graph = st.selectbox("Select Graph Type:",["Non Interactive", "Interactive"])

        val = st.slider("Filter data using years",0,11)
        data = data.loc[data['YearsExperience'] >= val]

        if graph == "Non Interactive":
            plt.figure(figsize=(10,5))
            plt.scatter(data['YearsExperience'], data['Salary'])
            plt.ylim(0)
            plt.xlabel('Years of Experience')
            plt.ylabel('Salary')
            plt.tight_layout()
            st.pyplot()

        if graph == "Interactive":
            layout = go.Layout(
                xaxis = dict(range=[0,11]),
                yaxis = dict(range=[0,120000])
            )
            fig = go.Figure(data = go.Scatter(x=data['YearsExperience'], y=data['Salary'], mode='markers'), layout=layout)
            st.plotly_chart(fig)
        
    if st.checkbox("Exploratory Data Analysis"):
        # Opening file from file path
        with open('./Salary_EDA.pdf', "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # Embedding PDF in HTML
        pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

        # Displaying File
        st.markdown(pdf_display, unsafe_allow_html=True)

if nav == "Salary Prediction":
    st.header("Predict your salary")
    
    x = np.array(data['YearsExperience']).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, np.array(data['Salary']))
    
    val = st.number_input("Enter Years of Experience", min_value=0.00, max_value=100.00, value=1.00, step=0.25)
    val = np.array(val).reshape(-1, 1)
    prediction = lr.predict(val)[0]
    
    if st.button("Predict") :
        st.success("Your predicted salary is: $%.2f" % (prediction))
    

if nav == "Contribute":
    st.header("Contribute to this dataset")
    experience = st.number_input("Enter years of experience", min_value=0.00, max_value=100.00, value=1.00, step=0.25, format="%.2f")
    salary = st.number_input("Enter salary", min_value=1000.00, max_value=8000000.00, value=10000.00, step=5000.00, format="%.2f")
    
    if st.button("Submit") :
        to_add = {'YearsExperience': experience, 'Salary': salary}
        to_add = pd.DataFrame(to_add, index=[0])
        to_add.to_csv('./Data/Salary_Data.csv', mode = 'a', index=False, header=False)
        st.success("Data added successfully!")
        data = pd.read_csv('./Data/Salary_Data.csv')
    
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    def verify_download():
        st.markdown("> :warning: **If your file hasn't downloaded after 10 seconds**: Click this [link](https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression)!")
    
    csv = convert_df(data)
    st.download_button(
        label="Download dataset as CSV",
        data=csv,
        file_name='Salary_Data.csv',
        mime='text/csv',
        on_click=verify_download
    )