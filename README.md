# Stroke Predictive and Prescriptive Analysis

This project performs a comprehensive analysis of stroke risk factors using machine learning techniques. It includes both predictive modeling and prescriptive analysis to identify key risk factors and provide insights for stroke prevention.

## Project Overview

The analysis focuses on various health and lifestyle factors that contribute to stroke risk, including:
- Age and gender
- Blood pressure (Systolic and Diastolic)
- Body Mass Index (BMI)
- Average Glucose Level
- Lifestyle factors (Smoking, Alcohol Intake, Stress Levels)
- Family History
- Symptoms (Blurred Vision, Confusion, Difficulty Speaking, Dizziness, Weakness)

## Repository Structure

- `analyze_stroke_data.py`: Main script for data analysis and visualization
- `detailed_stroke_analysis.py`: Advanced analysis and machine learning implementation
- `Stroke_Analysis_Report.docx`: Comprehensive report of findings
- Various visualization files (PNG format) showing:
  - Feature distributions
  - Correlation analysis
  - Model performance metrics
  - Risk factor analysis

## Key Visualizations

- Age distribution analysis
- BMI vs Glucose Level scatter plot
- Correlation matrix of risk factors
- Feature importance plots
- Confusion matrices for different models
- ROC curves for model evaluation

## Machine Learning Models

The project implements several machine learning models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

## Requirements

To run this project, you'll need:
- Python 3.x
- Required Python packages (listed in requirements.txt):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - openpyxl

## Setup Instructions

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
4. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the basic analysis:
   ```
   python analyze_stroke_data.py
   ```
2. For detailed analysis and machine learning models:
   ```
   python detailed_stroke_analysis.py
   ```

## Results

The analysis provides:
- Risk factor identification
- Model performance metrics
- Feature importance rankings
- Visual representations of relationships between variables
- Prescriptive insights for stroke prevention

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 