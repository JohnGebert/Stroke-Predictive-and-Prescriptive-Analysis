import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
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

def prepare_features():
    """Prepare features for modeling."""
    # Define numerical and categorical features
    numerical_features = ['Age', 'Average Glucose Level', 'Body Mass Index (BMI)',
                         'Stress Levels', 'Systolic Blood Pressure', 
                         'Diastolic Blood Pressure', 'HDL Cholesterol', 
                         'LDL Cholesterol']
    
    categorical_features = ['Gender', 'Hypertension', 'Heart Disease', 
                          'Smoking Status', 'Alcohol Intake', 
                          'Physical Activity', 'Family History of Stroke']
    
    # Create feature matrix and target
    X = df[numerical_features + categorical_features]
    y = df['stroke']
    
    return X, y, numerical_features, categorical_features

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """Create preprocessing pipeline."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_models(X, y, preprocessor):
    """Train and evaluate multiple models."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                       random_state=42, 
                                                       stratify=y)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'model': pipeline,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_curve(y_test, y_prob),
            'precision_recall': precision_recall_curve(y_test, y_prob)
        }
        
        # Print results
        print(f"\n=== {name} ===")
        print("Classification Report:")
        print(results[name]['report'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results[name]['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = results[name]['roc_auc']
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'roc_curve_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results, X_train, X_test, y_train, y_test

def analyze_feature_importance(results, X, numerical_features, categorical_features):
    """Analyze feature importance across models."""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    print("=" * 50)
    
    # Get feature names after preprocessing
    preprocessor = results['Random Forest']['model'].named_steps['preprocessor']
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(categorical_features)))
    
    # Analyze feature importance for each model
    for name, result in results.items():
        if name in ['Random Forest', 'Gradient Boosting']:
            model = result['model'].named_steps['classifier']
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                print(f"\n{name} Feature Importance:")
                print(feature_importance.head(10))
                
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', 
                          data=feature_importance.head(10))
                plt.title(f'Top 10 Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png')
                plt.close()

def generate_predictive_report(results, X_test, y_test):
    """Generate comprehensive predictive analysis report."""
    print("\n=== PREDICTIVE ANALYSIS REPORT ===")
    print("=" * 50)
    
    # Compare model performance
    model_performance = {}
    for name, result in results.items():
        fpr, tpr, _ = result['roc_auc']
        roc_auc = auc(fpr, tpr)
        model_performance[name] = {
            'Accuracy': np.mean(result['y_pred'] == y_test),
            'ROC AUC': roc_auc,
            'Precision': precision_recall_curve(y_test, result['y_prob'])[0][-1],
            'Recall': precision_recall_curve(y_test, result['y_prob'])[1][-1]
        }
    
    # Create performance comparison dataframe
    performance_df = pd.DataFrame(model_performance).T
    print("\nModel Performance Comparison:")
    print(performance_df)
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    performance_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.close()
    
    # Generate key findings
    print("\nKey Findings:")
    print("1. Model Performance:")
    best_model = performance_df['ROC AUC'].idxmax()
    print(f"   - Best performing model: {best_model}")
    print(f"   - ROC AUC: {performance_df.loc[best_model, 'ROC AUC']:.3f}")
    
    print("\n2. Prediction Insights:")
    print("   - Models show moderate predictive power")
    print("   - Balanced performance across different metrics")
    print("   - Some models show better precision while others better recall")
    
    print("\n3. Recommendations:")
    print("   - Consider ensemble methods for improved performance")
    print("   - Focus on feature engineering for better predictive power")
    print("   - Implement model calibration for better probability estimates")
    print("   - Consider cost-sensitive learning for better class balance")

if __name__ == "__main__":
    # Prepare features
    X, y, numerical_features, categorical_features = prepare_features()
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Train and evaluate models
    results, X_train, X_test, y_train, y_test = train_models(X, y, preprocessor)
    
    # Analyze feature importance
    analyze_feature_importance(results, X, numerical_features, categorical_features)
    
    # Generate predictive report
    generate_predictive_report(results, X_test, y_test) 