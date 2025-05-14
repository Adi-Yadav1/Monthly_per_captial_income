@echo off
echo MPCE Prediction Project
echo =====================

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Running data analysis and model training...
python mpce_analysis.py

echo.
echo Starting the Streamlit application...
streamlit run mpce_app.py

pause