import streamlit as st
import pandas as pd
import numpy as np

from src.data.data_generator import generate_classpass_data
from src.models.booking_prediction import BookingLikelihoodModel
from src.models.churn_model import ChurnModel
from src.pricing.dynamic_pricing import DynamicPricingEngine
from src.features.trend_detection import detect_trends

st.set_page_config(page_title = "ClassPass Optimmization Engine", layout = "wide")

st.title (" ClassPass Optimization Engine")

#Genertating The Dataset

@st.cache_data
def load_data():
    return generate_classpass_data / 10000
