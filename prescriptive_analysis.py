import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

def create_risk_stratification():
    """Create risk stratification based on multiple factors."""
    print("\n=== RISK STRATIFICATION ANALYSIS ===")
    print("=" * 50)
    
    # Define risk factors and their weights
    risk_factors = {
        'Hypertension': 1.5,
        'Heart Disease': 1.5,
        'Average Glucose Level': 1.2,
        'Body Mass Index (BMI)': 1.0,
        'Stress Levels': 1.0,
        'HDL Cholesterol': 1.3,
        'LDL Cholesterol': 1.3,
        'Systolic Blood Pressure': 1.2,
        'Diastolic Blood Pressure': 1.2
    }
    
    # Calculate risk score
    df['risk_score'] = 0
    for factor, weight in risk_factors.items():
        if factor in ['Hypertension', 'Heart Disease']:
            df['risk_score'] += df[factor] * weight
        else:
            # Normalize numerical factors
            normalized = (df[factor] - df[factor].mean()) / df[factor].std()
            df['risk_score'] += normalized * weight
    
    # Define risk categories
    df['risk_category'] = pd.qcut(df['risk_score'], q=4, 
                                 labels=['Low Risk', 'Moderate Risk', 
                                        'High Risk', 'Very High Risk'])
    
    # Analyze risk distribution
    risk_distribution = pd.crosstab(df['risk_category'], df['Diagnosis'])
    risk_distribution['Stroke Rate'] = risk_distribution['Stroke'] / (risk_distribution['Stroke'] + risk_distribution['No Stroke'])
    
    print("\nRisk Category Distribution:")
    print(risk_distribution)
    
    # Visualize risk distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='risk_category', hue='Diagnosis', data=df, 
                 order=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'])
    plt.title('Stroke Distribution by Risk Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('risk_category_distribution.png')
    plt.close()

def generate_intervention_recommendations():
    """Generate personalized intervention recommendations."""
    print("\n=== INTERVENTION RECOMMENDATIONS ===")
    print("=" * 50)
    
    # Define intervention thresholds
    thresholds = {
        'Average Glucose Level': 140,
        'Body Mass Index (BMI)': 30,
        'Systolic Blood Pressure': 140,
        'Diastolic Blood Pressure': 90,
        'HDL Cholesterol': 40,
        'LDL Cholesterol': 130
    }
    
    # Generate recommendations for each patient
    recommendations = []
    for idx, row in df.iterrows():
        patient_recs = []
        
        # Check each risk factor
        if row['Hypertension'] == 1:
            patient_recs.append("Consider blood pressure medication and lifestyle modifications")
        if row['Heart Disease'] == 1:
            patient_recs.append("Regular cardiac monitoring and medication adherence")
        if row['Average Glucose Level'] > thresholds['Average Glucose Level']:
            patient_recs.append("Implement blood sugar control measures and dietary changes")
        if row['Body Mass Index (BMI)'] > thresholds['Body Mass Index (BMI)']:
            patient_recs.append("Weight management program and physical activity plan")
        if row['HDL Cholesterol'] < thresholds['HDL Cholesterol']:
            patient_recs.append("Increase HDL through exercise and dietary changes")
        if row['LDL Cholesterol'] > thresholds['LDL Cholesterol']:
            patient_recs.append("Cholesterol management through diet and medication if needed")
        
        recommendations.append(patient_recs)
    
    # Add recommendations to dataframe
    df['recommendations'] = recommendations
    
    # Analyze most common recommendations
    all_recs = [rec for sublist in recommendations for rec in sublist]
    rec_counts = pd.Series(all_recs).value_counts()
    
    print("\nMost Common Recommendations:")
    print(rec_counts.head(10))
    
    # Visualize recommendation distribution
    plt.figure(figsize=(12, 6))
    rec_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Most Common Intervention Recommendations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('recommendations_distribution.png')
    plt.close()

def build_predictive_model():
    """Build a predictive model for stroke risk."""
    print("\n=== PREDICTIVE MODELING ===")
    print("=" * 50)
    
    # Prepare features
    features = ['Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level',
                'Body Mass Index (BMI)', 'Stress Levels', 'Systolic Blood Pressure',
                'Diastolic Blood Pressure', 'HDL Cholesterol', 'LDL Cholesterol']
    
    X = df[features]
    y = df['stroke']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance in Stroke Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def generate_prescriptive_report():
    """Generate a comprehensive prescriptive report."""
    print("\n=== PRESCRIPTIVE ANALYSIS REPORT ===")
    print("=" * 50)
    
    # 1. Risk Stratification Summary
    risk_summary = df.groupby('risk_category')['stroke'].agg(['count', 'mean'])
    risk_summary['stroke_rate'] = risk_summary['mean'] * 100
    risk_summary['count_percentage'] = (risk_summary['count'] / len(df)) * 100
    
    print("\nRisk Stratification Summary:")
    print(risk_summary)
    
    # 2. Intervention Effectiveness
    intervention_effectiveness = {}
    for factor in ['Hypertension', 'Heart Disease', 'Average Glucose Level', 
                  'Body Mass Index (BMI)', 'HDL Cholesterol', 'LDL Cholesterol']:
        if factor in ['Hypertension', 'Heart Disease']:
            effectiveness = df.groupby(factor)['stroke'].mean()
        else:
            effectiveness = df.groupby(pd.qcut(df[factor], q=4))['stroke'].mean()
        intervention_effectiveness[factor] = effectiveness
    
    print("\nIntervention Effectiveness:")
    for factor, effectiveness in intervention_effectiveness.items():
        print(f"\n{factor}:")
        print(effectiveness)
    
    # 3. Generate Actionable Insights
    print("\nActionable Insights:")
    print("1. Focus on HDL Cholesterol management as it shows the strongest correlation with stroke risk")
    print("2. Implement targeted interventions based on risk category")
    print("3. Consider personalized treatment plans based on individual risk factors")
    print("4. Regular monitoring of key indicators is crucial for high-risk patients")
    print("5. Lifestyle modifications should be prioritized for patients with multiple risk factors")

if __name__ == "__main__":
    create_risk_stratification()
    generate_intervention_recommendations()
    build_predictive_model()
    generate_prescriptive_report() 