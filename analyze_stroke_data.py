import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Read the Excel file
file_path = Path('Stroke_Prediction_Dataset.xlsx')
try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Error reading the Excel file: {e}")
    exit(1)

# Create a binary stroke column (1 for 'Stroke', 0 for 'No Stroke')
df['stroke'] = (df['Diagnosis'] == 'Stroke').astype(int)

# Display basic information about the dataset
print("\nDataset Information:")
print("=" * 50)
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Display stroke distribution
print("\nStroke Distribution:")
print("=" * 50)
print(df['Diagnosis'].value_counts())

# Basic statistics for numerical columns
print("\nBasic Statistics:")
print("=" * 50)
print(df.describe())

# Create visualizations
try:
    # 1. Distribution of stroke cases
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Diagnosis', data=df)
    plt.title('Distribution of Stroke Cases')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stroke_distribution.png')
    plt.close()

    # 2. Age distribution by diagnosis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Diagnosis', y='Age', data=df)
    plt.title('Age Distribution by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel('Age')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('age_distribution.png')
    plt.close()

    # 3. Correlation matrix for numerical variables
    numerical_cols = ['Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level', 
                     'Body Mass Index (BMI)', 'Stress Levels', 'Systolic Blood Pressure', 
                     'Diastolic Blood Pressure', 'HDL Cholesterol', 'LDL Cholesterol', 'stroke']
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    # 4. Key risk factors analysis
    risk_factors = ['Gender', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 
                   'Family History of Stroke', 'Dietary Habits']
    
    for factor in risk_factors:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=factor, hue='Diagnosis', data=df)
        plt.title(f'Distribution of {factor} by Diagnosis')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{factor}_distribution.png')
        plt.close()

    # 5. BMI vs Glucose Level by Diagnosis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Body Mass Index (BMI)', y='Average Glucose Level', 
                   hue='Diagnosis', alpha=0.6)
    plt.title('BMI vs Average Glucose Level by Diagnosis')
    plt.tight_layout()
    plt.savefig('bmi_glucose_scatter.png')
    plt.close()

    print("\nAnalysis complete! Check the generated plots for visual insights.")
    
    # Print key findings
    print("\nKey Findings:")
    print("=" * 50)
    print(f"Total number of patients: {len(df)}")
    print(f"Number of stroke cases: {sum(df['stroke'])}")
    print(f"Stroke rate: {(sum(df['stroke'])/len(df))*100:.2f}%")
    
    # Calculate average age for stroke vs no stroke
    avg_age_stroke = df[df['stroke'] == 1]['Age'].mean()
    avg_age_no_stroke = df[df['stroke'] == 0]['Age'].mean()
    print(f"\nAverage age of stroke patients: {avg_age_stroke:.1f} years")
    print(f"Average age of non-stroke patients: {avg_age_no_stroke:.1f} years")
    
    # Calculate correlation with stroke
    correlations = df[numerical_cols].corr()['stroke'].sort_values(ascending=False)
    print("\nTop correlations with stroke:")
    print(correlations)

except Exception as e:
    print(f"Error during visualization: {e}") 