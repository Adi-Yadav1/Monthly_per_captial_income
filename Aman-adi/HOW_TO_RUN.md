# How to Run the MPCE Prediction Project

This guide provides step-by-step instructions to run the Monthly Per Capita Expenditure (MPCE) Prediction project.

## Prerequisites

1. Python 3.7 or higher
2. Required Python packages (listed in `requirements.txt`)

## Installation Steps

1. **Install Required Packages**

   Open a command prompt or terminal and run:

   ```
   pip install -r requirements.txt
   ```

   This will install all the necessary packages for the project.

2. **Run the Analysis**

   You have two options:

   **Option 1**: Run the Python script directly
   ```
   python mpce_analysis.py
   ```

   **Option 2**: Open and run the Jupyter notebook
   ```
   jupyter notebook MPCE_Analysis.ipynb
   ```
   Then run all cells in the notebook.

   This step will:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Train and evaluate multiple models
   - Save the best model as `best_model.pkl`
   - Generate visualizations in the `plots/` directory

3. **Launch the Streamlit Application**

   ```
   streamlit run mpce_app.py
   ```

   This will start the web application and open it in your default browser.

## Alternative: Using the Batch File

For Windows users, you can simply run the provided batch file:

1. Double-click on `run_mpce_project.bat`
2. This will automatically install the required packages, run the analysis, and launch the Streamlit application.

## Troubleshooting

If you encounter any issues:

1. **Missing packages**: Ensure all packages are installed correctly
   ```
   pip install -r requirements.txt
   ```

2. **File not found errors**: Make sure you're running the commands from the project root directory

3. **Model loading errors**: Ensure you've run the analysis script before launching the Streamlit app

4. **Streamlit port issues**: If the default port is in use, Streamlit will automatically use a different port

## Project Structure

- `mpce_dataset.csv`: The dataset containing household information
- `mpce_analysis.py`: Python script for data analysis and model training
- `MPCE_Analysis.ipynb`: Jupyter notebook version of the analysis
- `mpce_app.py`: Streamlit application for the user interface
- `best_model.pkl`: Saved model file (generated after running the analysis)
- `plots/`: Directory containing visualizations (generated during analysis)
- `requirements.txt`: List of required Python packages
- `README.md`: Project overview and documentation
- `run_mpce_project.bat`: Batch file to automate the setup and execution (Windows only)