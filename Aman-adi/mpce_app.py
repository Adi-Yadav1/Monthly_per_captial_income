import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from PIL import Image

# Set dark theme for plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

# Custom plot styling for dark theme
plt.rcParams.update({
    'figure.facecolor': '#121212',
    'axes.facecolor': '#1E1E1E',
    'axes.edgecolor': '#FFFFFF',
    'axes.labelcolor': '#FFFFFF',
    'axes.titlecolor': '#FFFFFF',
    'xtick.color': '#FFFFFF',
    'ytick.color': '#FFFFFF',
    'grid.color': '#333333',
    'text.color': '#FFFFFF',
    'legend.facecolor': '#1E1E1E',
    'legend.edgecolor': '#333333',
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# Set page configuration
st.set_page_config(
    page_title="MPCE Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with black background and white text
st.markdown("""
<style>
    /* Override Streamlit's default styling */
    .stApp {
        background-color: #121212;
    }
    
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.8rem;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
        border-bottom: 2px solid #FFFFFF;
        padding-bottom: 0.5rem;
    }
    
    /* Section styling */
    .section {
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #333333;
        font-weight: bold;
    }
    
    /* Highlight sections */
    .highlight {
        background-color: #000000;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .highlight h3, .highlight h4 {
        color: #FFFFFF;
        font-weight: bold;
    }
    
    .highlight p, .highlight ul, .highlight li {
        color: #FFFFFF;
        font-weight: bold;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.2rem;
        font-size: 0.9rem;
        color: #FFFFFF;
        background-color: #000000;
        border-radius: 0.5rem;
        border-top: 1px solid #333333;
    }
    
    /* Improve contrast for all text */
    p, li {
        color: #FFFFFF;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    /* Custom card styles with better contrast */
    .info-card {
        background-color: #000000;
        color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .success-card {
        background-color: #000000;
        color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2E7D32;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .warning-card {
        background-color: #000000;
        color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #EF6C00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Make sure links are visible */
    a {
        color: #64B5F6;
        font-weight: bold;
        text-decoration: underline;
    }
    
    /* Ensure table text is readable */
    table {
        color: #FFFFFF;
        background-color: #1E1E1E;
    }
    
    /* Style for Streamlit widgets */
    .stTextInput, .stSelectbox, .stNumberInput {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Style for Streamlit dataframes */
    .dataframe {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Style for Streamlit buttons */
    .stButton>button {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
        font-weight: bold;
    }
    
    /* Style for Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E1E;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        font-weight: bold;
    }
    
    /* Style for Streamlit plots */
    .stPlot {
        background-color: #1E1E1E;
    }
    
    /* Style for Streamlit metrics */
    .stMetric {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('mpce_dataset.csv')
    return df

# Load the model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to get percentile rank
def get_percentile(df, mpce_value):
    return round(sum(df['MPCE'] <= mpce_value) / len(df) * 100, 2)

# Main function
def main():
    # Load data and model
    try:
        df = load_data()
        model = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        model_loaded = False
        df = pd.DataFrame()
    
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center; color: #FFFFFF; font-weight: bold; font-size: 1.8rem; border-bottom: 2px solid #FFFFFF; padding-bottom: 0.8rem; margin-bottom: 1.5rem;'>MPCE Prediction App</h2>", unsafe_allow_html=True)
    st.sidebar.image("https://www.india.gov.in/sites/upload_files/npi/files/spotlights/budget-2023-24-inner-banner.jpg", use_column_width=True)
    
    # Add a dark background to the sidebar
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333333;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: #FFFFFF;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "Predict MPCE", "Data Analysis", "About"])
    
    # Home Page
    if page == "Home":
        st.markdown("<h1 class='main-header'>Monthly Per Capita Expenditure (MPCE) Prediction</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class='highlight'>
            <h3 style="color: #FFFFFF; font-weight: bold; font-size: 1.8rem; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Welcome to the MPCE Prediction App!</h3>
            <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem; text-align: justify; line-height: 1.6;">This application helps predict the Monthly Per Capita Expenditure (MPCE) for Indian households based on various socio-economic factors. The predictions can assist in:</p>
            <ul style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem; line-height: 1.8;">
                <li>Government budget planning and execution</li>
                <li>Policy making for welfare schemes</li>
                <li>Understanding expenditure patterns across different demographics</li>
                <li>Identifying factors that influence household expenditure</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h2 class='sub-header'>What is MPCE?</h2>", unsafe_allow_html=True)
            st.markdown("""
            <div class='section'>
            <p style='color: #FFFFFF; font-size: 1.2rem; font-weight: bold;'>Monthly Per Capita Expenditure (MPCE) is a measure of the average monthly expenditure per person in a household. It is calculated by dividing the total household expenditure by the number of members in the household.</p>
            <p style='color: #FFFFFF; font-size: 1.2rem; font-weight: bold;'>MPCE is an important economic indicator used by the Government of India to:</p>
            <ul style='color: #FFFFFF; font-size: 1.2rem; font-weight: bold;'>
                <li>Measure living standards</li>
                <li>Determine poverty lines</li>
                <li>Allocate resources for welfare programs</li>
                <li>Track changes in consumption patterns over time</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if os.path.exists('plots/mpce_distribution.png'):
                st.image('plots/mpce_distribution.png', caption='MPCE Distribution', use_column_width=True)
            
            if os.path.exists('plots/model_comparison.png'):
                st.image('plots/model_comparison.png', caption='Model Performance Comparison', use_column_width=True)
        
        st.markdown("<h2 class='sub-header'>How to Use This App</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
            <h3 style="color: white; font-weight: bold;">Predict MPCE</h3>
            <p style="color: white;">Enter household details to get a prediction of the Monthly Per Capita Expenditure.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='success-card'>
            <h3 style="color: white; font-weight: bold;">Data Analysis</h3>
            <p style="color: white;">Explore visualizations and insights from the MPCE dataset.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='warning-card'>
            <h3 style="color: white; font-weight: bold;">About</h3>
            <p style="color: white;">Learn more about the project, methodology, and data sources.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Predict MPCE Page
    elif page == "Predict MPCE":
        st.markdown("<h1 class='main-header'>Predict Monthly Per Capita Expenditure</h1>", unsafe_allow_html=True)
        
        if not model_loaded:
            st.warning("Please run the analysis script first to generate the model.")
            return
        
        st.markdown("<p>Enter the household details below to predict the Monthly Per Capita Expenditure (MPCE).</p>", unsafe_allow_html=True)
        
        # Create form for user inputs
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                state = st.selectbox("State", sorted(df['State'].unique()))
                rural_urban = st.selectbox("Area Type", df['Rural_Urban'].unique())
                household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
                monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=200000, value=30000)
                education_level = st.selectbox("Education Level", sorted(df['Education_Level'].unique()))
                employment_status = st.selectbox("Employment Status", df['Employment_Status'].unique())
                age_household_head = st.number_input("Age of Household Head", min_value=18, max_value=100, value=40)
                gender_household_head = st.selectbox("Gender of Household Head", df['Gender_Household_Head'].unique())
                
            with col2:
                electricity_access = st.selectbox("Electricity Access", ["Yes", "No"])
                internet_access = st.selectbox("Internet Access", ["Yes", "No"])
                cooking_fuel_type = st.selectbox("Cooking Fuel Type", sorted(df['Cooking_Fuel_Type'].unique()))
                water_source = st.selectbox("Water Source", sorted(df['Water_Source'].unique()))
                sanitation_facility = st.selectbox("Sanitation Facility", ["Yes", "No"])
                vehicle_ownership = st.number_input("Number of Vehicles Owned", min_value=0, max_value=10, value=1)
                loan_repayment = st.number_input("Monthly Loan Repayment (₹)", min_value=0, max_value=50000, value=5000)
                savings = st.number_input("Savings (₹)", min_value=0, max_value=100000, value=10000)
                
            with col3:
                mobile_phones_owned = st.number_input("Mobile Phones Owned", min_value=0, max_value=10, value=2)
                tv_ownership = st.selectbox("TV Ownership", ["Yes", "No"])
                computer_ownership = st.selectbox("Computer Ownership", ["Yes", "No"])
                internet_data_usage = st.number_input("Internet Data Usage (GB)", min_value=0, max_value=1000, value=50)
                type_of_employment = st.selectbox("Type of Employment", sorted(df['Type_of_Employment'].unique()))
                agricultural_land_owned = st.number_input("Agricultural Land Owned (acres)", min_value=0, max_value=50, value=0)
                financial_aid_received = st.selectbox("Financial Aid Received", ["Yes", "No"])
                type_of_house = st.selectbox("Type of House", sorted(df['Type_of_House'].unique()))
                debt_amount = st.number_input("Debt Amount (₹)", min_value=0, max_value=1000000, value=50000)
                insurance_coverage = st.selectbox("Insurance Coverage", ["Yes", "No"])
            
            # Food and other expenditures
            st.markdown("<h3>Expenditure Details</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                food_expenditure = st.number_input("Food Expenditure (₹)", min_value=0, max_value=50000, value=10000)
            
            with col2:
                housing_expenditure = st.number_input("Housing Expenditure (₹)", min_value=0, max_value=50000, value=8000)
            
            with col3:
                transport_expenditure = st.number_input("Transport Expenditure (₹)", min_value=0, max_value=20000, value=3000)
            
            with col4:
                healthcare_expenditure = st.number_input("Healthcare Expenditure (₹)", min_value=0, max_value=20000, value=2000)
            
            with col5:
                other_expenditures = st.number_input("Other Expenditures (₹)", min_value=0, max_value=30000, value=5000)
            
            # Calculate total expenditure
            total_expenditure = (food_expenditure + housing_expenditure + transport_expenditure + 
                                healthcare_expenditure + other_expenditures)
            
            submit_button = st.form_submit_button("Predict MPCE")
        
        if submit_button:
            # Create a dataframe with the input values
            input_data = pd.DataFrame({
                'State': [state],
                'Rural_Urban': [rural_urban],
                'Household_Size': [household_size],
                'Monthly_Income': [monthly_income],
                'Education_Level': [education_level],
                'Employment_Status': [employment_status],
                'Food_Expenditure': [food_expenditure],
                'Housing_Expenditure': [housing_expenditure],
                'Transport_Expenditure': [transport_expenditure],
                'Healthcare_Expenditure': [healthcare_expenditure],
                'Other_Expenditures': [other_expenditures],
                'Total_Expenditure': [total_expenditure],
                'Age_Household_Head': [age_household_head],
                'Gender_Household_Head': [gender_household_head],
                'Electricity_Access': [electricity_access],
                'Internet_Access': [internet_access],
                'Cooking_Fuel_Type': [cooking_fuel_type],
                'Water_Source': [water_source],
                'Sanitation_Facility': [sanitation_facility],
                'Vehicle_Ownership': [vehicle_ownership],
                'Loan_Repayment': [loan_repayment],
                'Savings': [savings],
                'Mobile_Phones_Owned': [mobile_phones_owned],
                'TV_Ownership': [tv_ownership],
                'Computer_Ownership': [computer_ownership],
                'Internet_Data_Usage_GB': [internet_data_usage],
                'Type_of_Employment': [type_of_employment],
                'Agricultural_Land_Owned': [agricultural_land_owned],
                'Financial_Aid_Received': [financial_aid_received],
                'Type_of_House': [type_of_house],
                'Debt_Amount': [debt_amount],
                'Insurance_Coverage': [insurance_coverage]
            })
            
            # Make prediction
            predicted_mpce = model.predict(input_data)[0]
            calculated_mpce = total_expenditure / household_size
            
            # Calculate percentile
            percentile = get_percentile(df, predicted_mpce)
            
            # Display results
            st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='highlight'>
                <h3 style="color: #FFFFFF; font-weight: bold; font-size: 1.8rem; text-align: center; margin-bottom: 1rem;">Predicted MPCE: ₹{predicted_mpce:.2f}</h3>
                <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem; text-align: center;">This is the model's prediction based on all input factors.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='success-card'>
                <h4 style="color: #FFFFFF; font-weight: bold; font-size: 1.4rem; border-bottom: 1px solid #FFFFFF; padding-bottom: 0.5rem;">Percentile Rank: {percentile}%</h4>
                <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem;">Your predicted MPCE is higher than approximately {percentile}% of Indian households in our dataset.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='warning-card'>
                <h3 style="color: #FFFFFF; font-weight: bold; font-size: 1.6rem; border-bottom: 1px solid #FFFFFF; padding-bottom: 0.5rem;">Calculated MPCE: ₹{calculated_mpce:.2f}</h3>
                <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem;">This is calculated by dividing the total expenditure (₹{total_expenditure:.2f}) by the household size ({household_size}).</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Compare with state average
                state_avg = df[df['State'] == state]['MPCE'].mean()
                diff_percent = (predicted_mpce - state_avg) / state_avg * 100
                
                st.markdown(f"""
                <div class='info-card'>
                <h4 style="color: #FFFFFF; font-weight: bold; font-size: 1.4rem; border-bottom: 1px solid #FFFFFF; padding-bottom: 0.5rem;">Comparison with {state} Average</h4>
                <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem;">State Average MPCE: ₹{state_avg:.2f}</p>
                <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem;">Your predicted MPCE is {abs(diff_percent):.2f}% {'higher' if diff_percent >= 0 else 'lower'} than the {state} average.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Expenditure breakdown
            st.markdown("<h3>Expenditure Breakdown</h3>", unsafe_allow_html=True)
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 6))
            expenditure_labels = ['Food', 'Housing', 'Transport', 'Healthcare', 'Others']
            expenditure_values = [food_expenditure, housing_expenditure, transport_expenditure, 
                                healthcare_expenditure, other_expenditures]
            
            # Calculate percentages
            total = sum(expenditure_values)
            percentages = [val/total*100 for val in expenditure_values]
            
            # Create labels with percentages
            labels = [f'{label} (₹{val:,.0f}, {pct:.1f}%)' for label, val, pct in zip(expenditure_labels, expenditure_values, percentages)]
            
            ax.pie(expenditure_values, labels=labels, autopct='', startangle=90, shadow=False)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Household Expenditure Breakdown')
            
            st.pyplot(fig)
            
            # Recommendations based on the prediction
            st.markdown("<h3 class='sub-header'>Insights and Recommendations</h3>", unsafe_allow_html=True)
            
            # Compare expenditure percentages with typical values
            food_percent = food_expenditure / total_expenditure * 100
            housing_percent = housing_expenditure / total_expenditure * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #FFFFFF; font-weight: bold; font-size: 1.3rem; border-bottom: 1px solid #FFFFFF; padding-bottom: 0.5rem;'>Food & Housing Expenditure</h4>", unsafe_allow_html=True)
                
                if food_percent > 40:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>⚠️ Food Expenditure:</strong> Your food expenditure ({food_percent:.1f}%) is higher than the typical range (30-40% of total). Consider reviewing your food budget.</p>", unsafe_allow_html=True)
                elif food_percent < 20:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>ℹ️ Food Expenditure:</strong> Your food expenditure ({food_percent:.1f}%) is lower than the typical range (30-40% of total). Ensure adequate nutrition is maintained.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>✅ Food Expenditure:</strong> Your food expenditure ({food_percent:.1f}%) is within the typical range (30-40% of total).</p>", unsafe_allow_html=True)
                
                if housing_percent > 35:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>⚠️ Housing Expenditure:</strong> Your housing expenditure ({housing_percent:.1f}%) is higher than the typical range (20-35% of total). Consider if there are ways to reduce housing costs.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>✅ Housing Expenditure:</strong> Your housing expenditure ({housing_percent:.1f}%) is within or below the typical range (20-35% of total).</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='success-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #FFFFFF; font-weight: bold; font-size: 1.3rem; border-bottom: 1px solid #FFFFFF; padding-bottom: 0.5rem;'>Financial Health</h4>", unsafe_allow_html=True)
                
                # Savings recommendation
                savings_percent = savings / monthly_income * 100
                if savings_percent < 10:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>⚠️ Savings:</strong> Your savings are only {savings_percent:.1f}% of your monthly income, which is less than the recommended 10%. Consider increasing your savings if possible.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>✅ Savings:</strong> Great job! You're saving {savings_percent:.1f}% of your monthly income, which meets or exceeds the recommended 10%.</p>", unsafe_allow_html=True)
                
                # Debt recommendation
                debt_income_ratio = debt_amount / (monthly_income * 12) * 100
                if debt_income_ratio > 50:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>⚠️ Debt:</strong> Your debt is {debt_income_ratio:.1f}% of your annual income, which is relatively high. Consider debt reduction strategies.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: #FFFFFF; font-weight: bold; font-size: 1.1rem;'><strong>✅ Debt:</strong> Your debt level ({debt_income_ratio:.1f}% of annual income) is within a manageable range.</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Data Analysis Page
    elif page == "Data Analysis":
        st.markdown("<h1 class='main-header'>MPCE Data Analysis</h1>", unsafe_allow_html=True)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["MPCE Distribution", "Geographic Analysis", "Socioeconomic Factors", "Expenditure Patterns", "Custom Analysis"])
        
        with tab1:
            st.markdown("<h2 class='sub-header'>MPCE Distribution Analysis</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists('plots/mpce_distribution.png'):
                    st.image('plots/mpce_distribution.png', caption='MPCE Distribution', use_column_width=True)
                else:
                    # Create the plot if it doesn't exist
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df['MPCE'], kde=True, ax=ax)
                    ax.set_title('Distribution of Monthly Per Capita Expenditure (MPCE)')
                    ax.set_xlabel('MPCE')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                st.markdown("""
                <div class='highlight'>
                <h4>Key Insights:</h4>
                <ul>
                    <li>The distribution of MPCE shows the spread of household expenditure across the population</li>
                    <li>The shape indicates whether most households have similar expenditure patterns or if there's high inequality</li>
                    <li>Outliers represent households with extremely high or low expenditure</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Calculate and display statistics
                mpce_stats = df['MPCE'].describe()
                
                st.markdown("<h4>MPCE Statistics</h4>", unsafe_allow_html=True)
                st.write(f"**Mean MPCE:** ₹{mpce_stats['mean']:.2f}")
                st.write(f"**Median MPCE:** ₹{mpce_stats['50%']:.2f}")
                st.write(f"**Minimum MPCE:** ₹{mpce_stats['min']:.2f}")
                st.write(f"**Maximum MPCE:** ₹{mpce_stats['max']:.2f}")
                st.write(f"**Standard Deviation:** ₹{mpce_stats['std']:.2f}")
                
                # Calculate percentiles
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(df['MPCE'], percentiles)
                
                st.markdown("<h4>MPCE Percentiles</h4>", unsafe_allow_html=True)
                for p, v in zip(percentiles, percentile_values):
                    st.write(f"**{p}th Percentile:** ₹{v:.2f}")
                
                # Calculate inequality measures
                q75, q25 = np.percentile(df['MPCE'], [75, 25])
                iqr = q75 - q25
                
                st.markdown("<h4>Inequality Measures</h4>", unsafe_allow_html=True)
                st.write(f"**Interquartile Range (IQR):** ₹{iqr:.2f}")
                st.write(f"**90/10 Ratio:** {percentile_values[5]/percentile_values[0]:.2f}")
        
        with tab2:
            st.markdown("<h2 class='sub-header'>Geographic Analysis</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists('plots/mpce_by_state.png'):
                    st.image('plots/mpce_by_state.png', caption='MPCE by State', use_column_width=True)
                else:
                    # Create the plot if it doesn't exist
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.boxplot(x='State', y='MPCE', data=df, ax=ax)
                    ax.set_title('MPCE Distribution by State')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with col2:
                if os.path.exists('plots/mpce_by_rural_urban.png'):
                    st.image('plots/mpce_by_rural_urban.png', caption='MPCE by Rural/Urban', use_column_width=True)
                else:
                    # Create the plot if it doesn't exist
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x='Rural_Urban', y='MPCE', data=df, ax=ax)
                    ax.set_title('MPCE Distribution by Rural/Urban')
                    st.pyplot(fig)
            
            # State-wise average MPCE
            st.markdown("<h4>Average MPCE by State</h4>", unsafe_allow_html=True)
            
            state_mpce = df.groupby('State')['MPCE'].agg(['mean', 'median', 'std', 'count']).reset_index()
            state_mpce.columns = ['State', 'Mean MPCE', 'Median MPCE', 'Std Dev', 'Count']
            state_mpce = state_mpce.sort_values('Mean MPCE', ascending=False)
            
            # Format the values
            state_mpce['Mean MPCE'] = state_mpce['Mean MPCE'].round(2)
            state_mpce['Median MPCE'] = state_mpce['Median MPCE'].round(2)
            state_mpce['Std Dev'] = state_mpce['Std Dev'].round(2)
            
            st.dataframe(state_mpce, use_container_width=True)
            
            # Rural vs Urban analysis
            st.markdown("<h4>Rural vs Urban MPCE Comparison</h4>", unsafe_allow_html=True)
            
            rural_urban_mpce = df.groupby('Rural_Urban')['MPCE'].agg(['mean', 'median', 'std', 'count']).reset_index()
            rural_urban_mpce.columns = ['Area Type', 'Mean MPCE', 'Median MPCE', 'Std Dev', 'Count']
            
            # Format the values
            rural_urban_mpce['Mean MPCE'] = rural_urban_mpce['Mean MPCE'].round(2)
            rural_urban_mpce['Median MPCE'] = rural_urban_mpce['Median MPCE'].round(2)
            rural_urban_mpce['Std Dev'] = rural_urban_mpce['Std Dev'].round(2)
            
            st.dataframe(rural_urban_mpce, use_container_width=True)
            
            # State and Rural/Urban combined analysis
            st.markdown("<h4>MPCE by State and Rural/Urban Areas</h4>", unsafe_allow_html=True)
            
            # Create a pivot table
            pivot = pd.pivot_table(df, values='MPCE', index='State', columns='Rural_Urban', aggfunc='mean')
            pivot = pivot.round(2)
            
            st.dataframe(pivot, use_container_width=True)
            
            # Plot the pivot table
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot.plot(kind='bar', ax=ax)
            ax.set_title('Average MPCE by State and Rural/Urban Areas')
            ax.set_ylabel('Average MPCE (₹)')
            ax.set_xlabel('State')
            plt.legend(title='Area Type')
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.markdown("<h2 class='sub-header'>Socioeconomic Factors Analysis</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists('plots/mpce_by_education.png'):
                    st.image('plots/mpce_by_education.png', caption='MPCE by Education Level', use_column_width=True)
                else:
                    # Create the plot if it doesn't exist
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.boxplot(x='Education_Level', y='MPCE', data=df, ax=ax)
                    ax.set_title('MPCE Distribution by Education Level')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with col2:
                if os.path.exists('plots/mpce_by_employment.png'):
                    st.image('plots/mpce_by_employment.png', caption='MPCE by Employment Status', use_column_width=True)
                else:
                    # Create the plot if it doesn't exist
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.boxplot(x='Employment_Status', y='MPCE', data=df, ax=ax)
                    ax.set_title('MPCE Distribution by Employment Status')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Income vs MPCE
            st.markdown("<h4>Relationship between Income and MPCE</h4>", unsafe_allow_html=True)
            
            if os.path.exists('plots/mpce_vs_income.png'):
                st.image('plots/mpce_vs_income.png', caption='MPCE vs Monthly Income', use_column_width=True)
            else:
                # Create the plot if it doesn't exist
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='Monthly_Income', y='MPCE', data=df, ax=ax)
                ax.set_title('MPCE vs Monthly Income')
                st.pyplot(fig)
            
            # Calculate correlation
            income_mpce_corr = df['Monthly_Income'].corr(df['MPCE'])
            st.write(f"**Correlation between Monthly Income and MPCE:** {income_mpce_corr:.4f}")
            
            # Education level analysis
            st.markdown("<h4>MPCE by Education Level</h4>", unsafe_allow_html=True)
            
            education_mpce = df.groupby('Education_Level')['MPCE'].agg(['mean', 'median', 'std', 'count']).reset_index()
            education_mpce.columns = ['Education Level', 'Mean MPCE', 'Median MPCE', 'Std Dev', 'Count']
            education_mpce = education_mpce.sort_values('Mean MPCE', ascending=False)
            
            # Format the values
            education_mpce['Mean MPCE'] = education_mpce['Mean MPCE'].round(2)
            education_mpce['Median MPCE'] = education_mpce['Median MPCE'].round(2)
            education_mpce['Std Dev'] = education_mpce['Std Dev'].round(2)
            
            st.dataframe(education_mpce, use_container_width=True)
            
            # Employment status analysis
            st.markdown("<h4>MPCE by Employment Status</h4>", unsafe_allow_html=True)
            
            employment_mpce = df.groupby('Employment_Status')['MPCE'].agg(['mean', 'median', 'std', 'count']).reset_index()
            employment_mpce.columns = ['Employment Status', 'Mean MPCE', 'Median MPCE', 'Std Dev', 'Count']
            employment_mpce = employment_mpce.sort_values('Mean MPCE', ascending=False)
            
            # Format the values
            employment_mpce['Mean MPCE'] = employment_mpce['Mean MPCE'].round(2)
            employment_mpce['Median MPCE'] = employment_mpce['Median MPCE'].round(2)
            employment_mpce['Std Dev'] = employment_mpce['Std Dev'].round(2)
            
            st.dataframe(employment_mpce, use_container_width=True)
            
            # Household size analysis
            st.markdown("<h4>MPCE by Household Size</h4>", unsafe_allow_html=True)
            
            # Create a bar chart
            household_mpce = df.groupby('Household_Size')['MPCE'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Household_Size', y='MPCE', data=household_mpce, ax=ax)
            ax.set_title('Average MPCE by Household Size')
            ax.set_xlabel('Household Size')
            ax.set_ylabel('Average MPCE (₹)')
            st.pyplot(fig)
        
        with tab4:
            st.markdown("<h2 class='sub-header'>Expenditure Patterns Analysis</h2>", unsafe_allow_html=True)
            
            # Calculate average expenditure by category
            expenditure_cols = ['Food_Expenditure', 'Housing_Expenditure', 'Transport_Expenditure', 
                               'Healthcare_Expenditure', 'Other_Expenditures']
            
            avg_expenditure = df[expenditure_cols].mean()
            
            # Create a pie chart
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = ['Food', 'Housing', 'Transport', 'Healthcare', 'Others']
            ax.pie(avg_expenditure, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            plt.title('Average Expenditure Breakdown')
            st.pyplot(fig)
            
            st.markdown("<h4>Average Expenditure by Category</h4>", unsafe_allow_html=True)
            
            # Create a dataframe for display
            avg_exp_df = pd.DataFrame({
                'Category': labels,
                'Average Expenditure (₹)': avg_expenditure.values,
                'Percentage of Total (%)': (avg_expenditure / avg_expenditure.sum() * 100).values
            })
            
            # Format the values
            avg_exp_df['Average Expenditure (₹)'] = avg_exp_df['Average Expenditure (₹)'].round(2)
            avg_exp_df['Percentage of Total (%)'] = avg_exp_df['Percentage of Total (%)'].round(2)
            
            st.dataframe(avg_exp_df, use_container_width=True)
            
            # Expenditure patterns by state
            st.markdown("<h4>Expenditure Patterns by State</h4>", unsafe_allow_html=True)
            
            # Select a state for analysis
            selected_state = st.selectbox("Select a State", sorted(df['State'].unique()), key="state_expenditure")
            
            # Filter data for the selected state
            state_data = df[df['State'] == selected_state]
            
            # Calculate average expenditure for the state
            state_avg_expenditure = state_data[expenditure_cols].mean()
            
            # Create a comparison bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create a dataframe for plotting
            comparison_data = pd.DataFrame({
                'Category': labels,
                'State Average': state_avg_expenditure.values,
                'National Average': avg_expenditure.values
            })
            
            # Reshape for plotting
            plot_data = pd.melt(comparison_data, id_vars='Category', var_name='Average Type', value_name='Amount')
            
            # Create the plot
            sns.barplot(x='Category', y='Amount', hue='Average Type', data=plot_data, ax=ax)
            ax.set_title(f'Expenditure Comparison: {selected_state} vs National Average')
            ax.set_ylabel('Amount (₹)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Expenditure patterns by income group
            st.markdown("<h4>Expenditure Patterns by Income Group</h4>", unsafe_allow_html=True)
            
            # Create income groups
            df['Income_Group'] = pd.qcut(df['Monthly_Income'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Calculate average expenditure by income group
            income_expenditure = df.groupby('Income_Group')[expenditure_cols].mean()
            
            # Calculate percentages
            income_expenditure_pct = income_expenditure.div(income_expenditure.sum(axis=1), axis=0) * 100
            
            # Create a stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            income_expenditure_pct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Expenditure Pattern by Income Group')
            ax.set_xlabel('Income Group')
            ax.set_ylabel('Percentage of Total Expenditure')
            plt.legend(title='Expenditure Category')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show the actual values
            st.markdown("<h5>Average Expenditure by Income Group (₹)</h5>", unsafe_allow_html=True)
            st.dataframe(income_expenditure.round(2), use_container_width=True)
            
            st.markdown("<h5>Expenditure Percentage by Income Group (%)</h5>", unsafe_allow_html=True)
            st.dataframe(income_expenditure_pct.round(2), use_container_width=True)
        
        with tab5:
            st.markdown("<h2 class='sub-header'>Custom Analysis</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <p>Create your own custom analysis by selecting variables to explore their relationship with MPCE.</p>
            """, unsafe_allow_html=True)
            
            # Select variables for analysis
            col1, col2 = st.columns(2)
            
            with col1:
                x_variable = st.selectbox("Select X Variable", df.columns.drop('MPCE'), key="custom_x")
                plot_type = st.selectbox("Select Plot Type", ["Box Plot", "Scatter Plot", "Bar Plot", "Violin Plot"], key="custom_plot")
            
            with col2:
                hue_variable = st.selectbox("Select Grouping Variable (Optional)", 
                                          ["None"] + list(df.select_dtypes(include=['object']).columns), 
                                          key="custom_hue")
                
                # For numerical X variables, allow binning
                if df[x_variable].dtype in ['int64', 'float64'] and plot_type in ["Box Plot", "Bar Plot"]:
                    bin_option = st.checkbox("Bin numerical variable", key="bin_option")
                    if bin_option:
                        num_bins = st.slider("Number of bins", 2, 10, 5, key="num_bins")
                        # Create bins
                        df['Binned_Variable'] = pd.qcut(df[x_variable], num_bins)
                        x_variable = 'Binned_Variable'
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if hue_variable == "None":
                hue = None
            else:
                hue = hue_variable
            
            if plot_type == "Box Plot":
                if df[x_variable].dtype in ['int64', 'float64'] and not 'Binned_Variable' in df.columns:
                    st.warning("Box plots work better with categorical X variables. Consider using binning or selecting a different plot type.")
                sns.boxplot(x=x_variable, y='MPCE', hue=hue, data=df, ax=ax)
            
            elif plot_type == "Scatter Plot":
                if df[x_variable].dtype not in ['int64', 'float64']:
                    st.warning("Scatter plots work better with numerical X variables. Consider selecting a different plot type.")
                sns.scatterplot(x=x_variable, y='MPCE', hue=hue, data=df, ax=ax)
            
            elif plot_type == "Bar Plot":
                if df[x_variable].dtype in ['int64', 'float64'] and not 'Binned_Variable' in df.columns:
                    # For numerical variables, calculate average MPCE
                    if hue is None:
                        grouped_data = df.groupby(pd.qcut(df[x_variable], 5))['MPCE'].mean().reset_index()
                        sns.barplot(x=x_variable, y='MPCE', data=grouped_data, ax=ax)
                    else:
                        st.warning("For numerical X variables with grouping, consider using binning.")
                else:
                    # For categorical variables
                    sns.barplot(x=x_variable, y='MPCE', hue=hue, data=df, ax=ax)
            
            elif plot_type == "Violin Plot":
                if df[x_variable].dtype in ['int64', 'float64'] and not 'Binned_Variable' in df.columns:
                    st.warning("Violin plots work better with categorical X variables. Consider using binning or selecting a different plot type.")
                sns.violinplot(x=x_variable, y='MPCE', hue=hue, data=df, ax=ax)
            
            ax.set_title(f'Relationship between {x_variable} and MPCE')
            ax.set_ylabel('MPCE (₹)')
            
            if df[x_variable].dtype == 'object' or 'Binned_Variable' in df.columns:
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate statistics
            st.markdown("<h4>Statistical Summary</h4>", unsafe_allow_html=True)
            
            if df[x_variable].dtype in ['int64', 'float64'] and not 'Binned_Variable' in df.columns:
                # For numerical variables, calculate correlation
                correlation = df[x_variable].corr(df['MPCE'])
                st.write(f"**Correlation between {x_variable} and MPCE:** {correlation:.4f}")
            else:
                # For categorical variables, show average MPCE by category
                if hue is None:
                    group_stats = df.groupby(x_variable)['MPCE'].agg(['mean', 'median', 'std', 'count']).reset_index()
                    group_stats.columns = [x_variable, 'Mean MPCE', 'Median MPCE', 'Std Dev', 'Count']
                    
                    # Format the values
                    group_stats['Mean MPCE'] = group_stats['Mean MPCE'].round(2)
                    group_stats['Median MPCE'] = group_stats['Median MPCE'].round(2)
                    group_stats['Std Dev'] = group_stats['Std Dev'].round(2)
                    
                    st.dataframe(group_stats.sort_values('Mean MPCE', ascending=False), use_container_width=True)
                else:
                    # For grouped analysis
                    group_stats = df.groupby([x_variable, hue])['MPCE'].mean().reset_index()
                    group_stats.columns = [x_variable, hue, 'Mean MPCE']
                    
                    # Format the values
                    group_stats['Mean MPCE'] = group_stats['Mean MPCE'].round(2)
                    
                    st.dataframe(group_stats.sort_values(['Mean MPCE'], ascending=False), use_container_width=True)
            
            # Clean up temporary columns
            if 'Binned_Variable' in df.columns:
                df.drop('Binned_Variable', axis=1, inplace=True)
    
    # About Page
    elif page == "About":
        st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3 style="color: #FFFFFF; font-weight: bold; font-size: 1.8rem; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Monthly Per Capita Expenditure (MPCE) Prediction Project</h3>
        <p style="color: #FFFFFF; font-weight: bold; font-size: 1.2rem; text-align: justify; line-height: 1.6;">This project aims to predict the Monthly Per Capita Expenditure (MPCE) for Indian households based on various socio-economic factors. The predictions can assist in government budget planning, policy making, and understanding expenditure patterns across different demographics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Project Methodology</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <ol>
            <li><strong>Data Collection:</strong> The dataset contains information about 1000 Indian households, including demographic details, income, expenditure patterns, and various socio-economic indicators.</li>
            <li><strong>Data Preprocessing:</strong> The data was cleaned by handling missing values, removing duplicates, and validating the MPCE calculations.</li>
            <li><strong>Exploratory Data Analysis (EDA):</strong> Various visualizations and statistical analyses were performed to understand the relationships between different variables and MPCE.</li>
            <li><strong>Feature Engineering:</strong> Relevant features were selected and transformed to improve model performance.</li>
            <li><strong>Model Selection:</strong> Multiple regression models were trained and evaluated, including Linear Regression, Random Forest, KNN, Support Vector Regression, and XGBoost.</li>
            <li><strong>Model Evaluation:</strong> Models were compared based on R² score, RMSE, and MAE to select the best performing model.</li>
            <li><strong>Deployment:</strong> The best model was deployed in this interactive web application to provide MPCE predictions and insights.</li>
        </ol>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Key Findings</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h4>Factors Influencing MPCE:</h4>
            <ul>
                <li>Monthly income has a strong positive correlation with MPCE</li>
                <li>Education level significantly impacts expenditure patterns</li>
                <li>Urban households generally have higher MPCE than rural households</li>
                <li>Household size has an inverse relationship with MPCE</li>
                <li>Employment status and type affect spending capacity</li>
            </ul>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <h4>Regional Variations:</h4>
            <ul>
                <li>Significant differences in MPCE across states</li>
                <li>Urban-rural divide varies by state</li>
                <li>Expenditure patterns differ based on regional factors</li>
                <li>Access to facilities (electricity, internet, water) impacts MPCE</li>
                <li>Housing type and ownership status influence expenditure</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Applications</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='success-card'>
        <h4 style="color: white; font-weight: bold;">Government and Policy Making:</h4>
        <ul style="color: white;">
            <li>Budget planning and resource allocation</li>
            <li>Designing targeted welfare schemes</li>
            <li>Evaluating the impact of economic policies</li>
            <li>Identifying vulnerable populations</li>
            <li>Monitoring changes in living standards over time</li>
        </ul>
        </div>
        
        <div class='info-card'>
        <h4 style="color: white; font-weight: bold;">Individual Households:</h4>
        <ul style="color: white;">
            <li>Benchmarking household expenditure against similar demographics</li>
            <li>Financial planning and budgeting</li>
            <li>Understanding expenditure patterns</li>
            <li>Identifying areas for potential savings</li>
            <li>Making informed financial decisions</li>
        </ul>
        </div>
        
        <div class='warning-card'>
        <h4 style="color: white; font-weight: bold;">Researchers and Analysts:</h4>
        <ul style="color: white;">
            <li>Studying consumption patterns and trends</li>
            <li>Analyzing economic inequality</li>
            <li>Conducting comparative studies across regions</li>
            <li>Developing economic indicators</li>
            <li>Forecasting future expenditure trends</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Data Sources and References</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>The dataset used in this project contains information about Indian households, including demographic details, income, expenditure patterns, and various socio-economic indicators.</p>
        
        <p>Key references and resources:</p>
        <ul>
            <li>National Sample Survey Office (NSSO) methodology for household consumption expenditure surveys</li>
            <li>Ministry of Statistics and Programme Implementation (MOSPI) guidelines</li>
            <li>Reserve Bank of India (RBI) reports on household finance</li>
            <li>World Bank standards for measuring living standards</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p style="font-weight: bold; color: #FFFFFF; font-size: 1.1rem;">MPCE Prediction App | Developed for Government of India Budget Planning and Execution</p>
        <p style="color: #FFFFFF; font-weight: bold;">© 2023 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()