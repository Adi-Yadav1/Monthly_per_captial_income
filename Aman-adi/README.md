# Monthly Per Capita Expenditure (MPCE) Prediction Project

## Project Overview
This project aims to predict the Monthly Per Capita Expenditure (MPCE) for Indian households based on various socio-economic factors. The predictions can assist in government budget planning, policy making, and understanding expenditure patterns across different demographics.

## Importance of MPCE
Monthly Per Capita Expenditure (MPCE) is a measure of the average monthly expenditure per person in a household. It is calculated by dividing the total household expenditure by the number of members in the household. MPCE is an important economic indicator used by the Government of India to:
- Measure living standards
- Determine poverty lines
- Allocate resources for welfare programs
- Track changes in consumption patterns over time

## Project Structure
The project is organized as follows:
1. **Data Collection**: Dataset containing information about Indian households
2. **Data Preprocessing**: Cleaning and preparing the data for analysis
3. **Exploratory Data Analysis (EDA)**: Understanding the relationships between variables
4. **Feature Engineering**: Selecting and transforming relevant features
5. **Model Selection**: Training and evaluating multiple regression models
6. **Model Deployment**: Creating an interactive web application for predictions

## Files in the Repository
- `mpce_dataset.csv`: The dataset containing household information
- `mpce_analysis.py`: Python script for data analysis and model training
- `mpce_app.py`: Streamlit application for the user interface
- `best_model.pkl`: Saved model file (generated after running the analysis script)
- `plots/`: Directory containing visualizations (generated during analysis)

## How to Run the Project

### Prerequisites
- Python 3.7+
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, streamlit

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### Running the Analysis
```bash
python mpce_analysis.py
```
This will:
- Load and preprocess the data
- Perform exploratory data analysis
- Train and evaluate multiple models
- Save the best model as `best_model.pkl`
- Generate visualizations in the `plots/` directory

### Running the Streamlit App
```bash
streamlit run mpce_app.py
```
This will launch the web application with the following features:
- MPCE prediction based on user inputs
- Data analysis and visualizations
- Insights and recommendations

## Models Evaluated
1. Linear Regression
2. Random Forest Regressor
3. K-Nearest Neighbors Regressor
4. Support Vector Regressor
5. XGBoost Regressor

The models are compared based on R² score, RMSE, and MAE to select the best performing model.

## Key Features of the Application
- **Predict MPCE**: Enter household details to get a prediction of the Monthly Per Capita Expenditure
- **Data Analysis**: Explore visualizations and insights from the MPCE dataset
- **Expenditure Breakdown**: Analyze how expenditure is distributed across different categories
- **Geographic Analysis**: Compare MPCE across states and rural/urban areas
- **Socioeconomic Factors**: Understand how education, employment, and other factors affect MPCE
- **Custom Analysis**: Create your own visualizations to explore relationships between variables

## Applications
- **Government and Policy Making**: Budget planning, resource allocation, designing welfare schemes
- **Individual Households**: Benchmarking expenditure, financial planning, identifying savings opportunities
- **Researchers and Analysts**: Studying consumption patterns, analyzing economic inequality, forecasting trends

## Future Enhancements
- Integration with real-time economic indicators
- Addition of time-series analysis for trend forecasting
- Incorporation of geospatial data for more detailed regional analysis
- Development of personalized financial recommendations
- Extension to mobile platforms for wider accessibility

## Acknowledgements
- National Sample Survey Office (NSSO) methodology for household consumption expenditure surveys
- Ministry of Statistics and Programme Implementation (MOSPI) guidelines
- Reserve Bank of India (RBI) reports on household finance
- World Bank standards for measuring living standards