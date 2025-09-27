import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pydeck as pdk

# ----------------------
# Load Model and Data
# ----------------------
gb_model_path = r"D:\SIT720\distinction_task\model\gradient_boosting_model.pkl"
gb_pipeline = joblib.load(gb_model_path)

dataset_path = r"D:\SIT720\distinction_task\melb_data.csv"
df = pd.read_csv(dataset_path)

st.set_page_config(page_title="Melbourne Housing Price Prediction", layout="wide")
st.title("🏠 Melbourne Housing Price Prediction")
st.sidebar.header("Input Features")

# ----------------------
# User Inputs
# ----------------------
Rooms = st.sidebar.number_input("Rooms", 1, 10, 3)
Distance = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 10.0)
Bathroom = st.sidebar.number_input("Bathroom", 1, 10, 2)
Car = st.sidebar.number_input("Car", 0, 10, 2)
Landsize = st.sidebar.number_input("Landsize", 0.0, 5000.0, 159.0)
BuildingArea = st.sidebar.number_input("Building Area", 0.0, 1000.0, 150.0)
Method = st.sidebar.selectbox("Method", df['Method'].dropna().unique())
SellerG = st.sidebar.selectbox("SellerG", df['SellerG'].dropna().unique())
Suburb = st.sidebar.selectbox("Suburb", df['Suburb'].dropna().unique())
Type = st.sidebar.selectbox("Type", df['Type'].dropna().unique())
Regionname = st.sidebar.selectbox("Region Name", df['Regionname'].dropna().unique())
CouncilArea = st.sidebar.selectbox("Council Area", df['CouncilArea'].dropna().unique())

input_data = pd.DataFrame([{
    'Rooms': Rooms,
    'Distance': Distance,
    'Bathroom': Bathroom,
    'Car': Car,
    'Landsize': Landsize,
    'BuildingArea': BuildingArea,
    'Method': Method,
    'SellerG': SellerG,
    'Suburb': Suburb,
    'Type': Type,
    'Regionname': Regionname,
    'CouncilArea': CouncilArea
}])

# ----------------------
# Prediction
# ----------------------
st.subheader("🏷 Price Prediction")
if st.button("Predict"):
    gb_pred = gb_pipeline.predict(input_data)[0]
    ci_gb = (gb_pred * 0.9, gb_pred * 1.1)
    
    st.success(f"Gradient Boosting Prediction: ${gb_pred:,.2f} (±10%: ${ci_gb[0]:,.2f} - ${ci_gb[1]:,.2f})")

# ----------------------
# Data Filters
# ----------------------
st.sidebar.header("Filter Dataset")
price_range = st.sidebar.slider(
    "Price Range", 
    int(df['Price'].min()), 
    int(df['Price'].max()), 
    (int(df['Price'].min()), int(df['Price'].max()))
)
filtered_df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]

# ----------------------
# Dataset Visualizations
# ----------------------
st.header("📊 Dataset Visualizations")

# Price Distribution
st.subheader("Price Distribution")
fig = px.histogram(filtered_df, x='Price', nbins=50, marginal='box', color_discrete_sequence=['skyblue'])
st.plotly_chart(fig, use_container_width=True)

# Rooms vs Price (Fixed ✅)
st.subheader("Rooms vs Price")
fig = px.box(
    filtered_df,
    x='Rooms',
    y='Price',
    color='Rooms',  # categorical coloring is fine
    title="Price Distribution by Number of Rooms"
)
st.plotly_chart(fig, use_container_width=True)

# Distance vs Price
st.subheader("Distance vs Price")
fig = px.scatter(
    filtered_df,
    x='Distance',
    y='Price',
    color='Rooms',
    size='Landsize',
    hover_data=['Suburb'],
    title="Distance vs Price Relationship"
)
st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = filtered_df.select_dtypes(include=np.number).corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature Importance (Gradient Boosting)
st.subheader("Feature Importance (Gradient Boosting)")
try:
    importances = gb_pipeline.named_steps['gradientboostingregressor'].feature_importances_
    features = input_data.columns
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
except Exception as e:
    st.info("Feature importance not available for this pipeline.")

# Map Visualization
st.subheader("Houses Map")
if 'Lattitude' in filtered_df.columns and 'Longtitude' in filtered_df.columns:
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=filtered_df['Lattitude'].mean(),
            longitude=filtered_df['Longtitude'].mean(),
            zoom=10,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_df,
                get_position='[Longtitude, Lattitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
                pickable=True
            )
        ]
    ))
else:
    st.info("Map not available: 'Lattitude' or 'Longtitude' columns missing.")

# Download Filtered Dataset
st.subheader("Download Filtered Dataset")
csv = filtered_df.to_csv(index=False)
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
