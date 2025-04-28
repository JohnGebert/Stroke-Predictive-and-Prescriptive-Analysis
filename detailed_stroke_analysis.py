import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set the style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Read the Excel file
file_path = Path('Stroke_Prediction_Dataset.xlsx')
df = pd.read_excel(file_path)
df['stroke'] = (df['Diagnosis'] == 'Stroke').astype(int)

def perform_chi_square_test(df, categorical_var):
    """Perform chi-square test for independence between categorical variable and stroke."""
    contingency_table = pd.crosstab(df[categorical_var], df['Diagnosis'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p

def perform_t_test(df, numerical_var):
    """Perform t-test for numerical variable between stroke and non-stroke groups."""
    stroke_group = df[df['stroke'] == 1][numerical_var]
    no_stroke_group = df[df['stroke'] == 0][numerical_var]
    t_stat, p_val = stats.ttest_ind(stroke_group, no_stroke_group)
    return t_stat, p_val

def analyze_risk_factors():
    """Analyze and visualize risk factors with statistical tests."""
    print("\n=== RISK FACTOR ANALYSIS ===")
    print("=" * 50)
    
    # 1. Categorical Risk Factors Analysis
    categorical_factors = ['Gender', 'Smoking Status', 'Alcohol Intake', 
                         'Physical Activity', 'Family History of Stroke', 
                         'Hypertension', 'Heart Disease']
    
    print("\nCategorical Risk Factors Analysis:")
    print("-" * 50)
    for factor in categorical_factors:
        chi2, p = perform_chi_square_test(df, factor)
        print(f"\n{factor}:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        print("Significant" if p < 0.05 else "Not significant")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.countplot(x=factor, hue='Diagnosis', data=df)
        plt.title(f'{factor} Distribution by Diagnosis')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{factor}_analysis.png')
        plt.close()

    # 2. Numerical Risk Factors Analysis
    numerical_factors = ['Age', 'Average Glucose Level', 'Body Mass Index (BMI)',
                        'Stress Levels', 'Systolic Blood Pressure', 
                        'Diastolic Blood Pressure', 'HDL Cholesterol', 
                        'LDL Cholesterol']
    
    print("\nNumerical Risk Factors Analysis:")
    print("-" * 50)
    for factor in numerical_factors:
        t_stat, p = perform_t_test(df, factor)
        print(f"\n{factor}:")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p:.4f}")
        print("Significant" if p < 0.05 else "Not significant")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Diagnosis', y=factor, data=df)
        plt.title(f'{factor} Distribution by Diagnosis')
        plt.tight_layout()
        plt.savefig(f'{factor}_analysis.png')
        plt.close()

    # 3. Combined Risk Factor Analysis
    print("\nCombined Risk Factor Analysis:")
    print("-" * 50)
    
    # Create a risk score based on significant factors
    risk_factors = ['Hypertension', 'Heart Disease', 'Average Glucose Level', 
                   'Body Mass Index (BMI)', 'Stress Levels']
    
    df['risk_score'] = 0
    for factor in risk_factors:
        if factor in numerical_factors:
            # Normalize numerical factors
            df['risk_score'] += (df[factor] - df[factor].mean()) / df[factor].std()
        else:
            # Add binary factors
            df['risk_score'] += df[factor]
    
    # Plot risk score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='risk_score', hue='Diagnosis', bins=30, 
                stat='density', common_norm=False)
    plt.title('Risk Score Distribution by Diagnosis')
    plt.tight_layout()
    plt.savefig('risk_score_distribution.png')
    plt.close()

    # 4. Symptom Analysis
    symptoms = ['Blurred Vision', 'Dizziness', 'Seizures', 'Weakness', 
               'Severe Fatigue', 'Headache', 'Confusion', 'Difficulty Speaking', 
               'Numbness', 'Loss of Balance']
    
    print("\nSymptom Analysis:")
    print("-" * 50)
    for symptom in symptoms:
        chi2, p = perform_chi_square_test(df, symptom)
        print(f"\n{symptom}:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        print("Significant" if p < 0.05 else "Not significant")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.countplot(x=symptom, hue='Diagnosis', data=df)
        plt.title(f'{symptom} Distribution by Diagnosis')
        plt.tight_layout()
        plt.savefig(f'{symptom}_analysis.png')
        plt.close()

    # 5. Generate Summary Report
    print("\n=== SUMMARY REPORT ===")
    print("=" * 50)
    print(f"Total patients analyzed: {len(df)}")
    print(f"Stroke cases: {sum(df['stroke'])}")
    print(f"Non-stroke cases: {len(df) - sum(df['stroke'])}")
    
    # Calculate odds ratios for significant factors
    print("\nSignificant Risk Factors (p < 0.05):")
    for factor in categorical_factors + numerical_factors:
        if factor in categorical_factors:
            chi2, p = perform_chi_square_test(df, factor)
        else:
            t_stat, p = perform_t_test(df, factor)
        
        if p < 0.05:
            print(f"\n{factor}:")
            if factor in categorical_factors:
                odds_ratio = pd.crosstab(df[factor], df['stroke']).apply(lambda x: x/x.sum(), axis=1)
                print(f"Odds Ratio: {odds_ratio.iloc[1,1]/odds_ratio.iloc[0,1]:.2f}")
            else:
                stroke_mean = df[df['stroke'] == 1][factor].mean()
                no_stroke_mean = df[df['stroke'] == 0][factor].mean()
                print(f"Mean difference: {stroke_mean - no_stroke_mean:.2f}")

if __name__ == "__main__":
    analyze_risk_factors() 