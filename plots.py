import numpy as np 
import pandas as pd 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt

chart_data = pd.DataFrame(np.random.randn(20,3), columns = ['Line 1 ' , 'Line 2 ' , 'Line 3'])

st.header('1. Chart with randm numbers')
st.subheader('1.1 Line Chart')
st.line_chart(chart_data)

st.subheader('1.2 Area Chart')
st.area_chart(chart_data)

st.subheader('1.3 Bar Graph')
st.bar_chart(chart_data)

st.header('2. Visualization with Matplotlib and Seaborn')

from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(data = iris.data , columns=iris.feature_names)

df['target']  = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])


st.subheader('2.1 Loading Dataset')

st.dataframe(df)

st.subheader('2.2 Bar Graph with Matplotlib')
fig = plt.figure(figsize=(15,8))
df['target_name'].value_counts().plot(kind='bar')
st.pyplot(fig)

st.subheader('2.3 Distribution Plot with Seaborn')
fig = plt.figure(figsize=(15,8))
sns.distplot(df['sepal length (cm)'])
st.pyplot(fig)

st.header('3. Multiple Graphs in one column')

col1 , col2 = st.columns(2)

with col1:
    col1.header = 'KDE = FALSE'
    fig1 = plt.figure(figsize=(5,5))
    sns.distplot(df['sepal length (cm)'] , kde=False)
    st.pyplot(fig1)

with col2:
    col2.header = 'Hist = False'
    fig2 = plt.figure(figsize=(5, 5))
    sns.distplot(df['sepal length (cm)'] , hist = False)
    st.pyplot(fig2)
    
st.header('4. Changing Style')
col4 , col5 , col6 = st.columns(3)

with col4:
    figx = plt.figure(figsize=(5,5))
    sns.set_style('darkgrid')
    sns.set_context('notebook')
    sns.distplot(df['sepal width (cm)'])
    st.pyplot(figx)

with col5:
    figx = plt.figure(figsize=(5,5))
    sns.set_style('dark')
    sns.set_context('notebook')
    sns.distplot(df['sepal length (cm)'])
    st.pyplot(figx)
    
with col6:
    figx = plt.figure(figsize=(5,5))
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.distplot(df['petal length (cm)'])
    st.pyplot(figx)
    

st.header('5. Exploring Different Graphs')
st.subheader('5.1 Scatter Plot')
fig , ax = plt.subplots(figsize=(15,8))
ax.scatter(*np.random.random(size = (2,100)))
st.pyplot(fig)

st.subheader('5.2 Count Plot')
fig= plt.figure(figsize=(15,8))
sns.countplot(data = df , x = 'target_name')
st.pyplot(fig)

st.subheader('5.3 Box Plot')
fig= plt.figure(figsize=(15,8))
sns.boxplot(data = df , x = 'target_name' , y = 'sepal width (cm)')
st.pyplot(fig)

st.subheader('5.3 Violin Plot')
fig= plt.figure(figsize=(15,8))
sns.violinplot(data = df , x = 'target_name' , y = 'sepal length (cm)')
st.pyplot(fig)
