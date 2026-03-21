# credit_card_default_app_complete.py
import os
# ===================== FILE PATHS =====================
RAW_DATA_PATH = 'UCI_Credit_Card.csv'
CLEAN_DATA_PATH = 'cleaned_data.csv'
MODEL_PATH = 'trained_models.pkl'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# RandomForest - keeping only Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, roc_auc_score)

# XGBoost - keeping only XGBoost
from xgboost import XGBClassifier

# Set page config
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #10B981;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .feature-importance-bar {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'training_status' not in st.session_state:
    st.session_state.training_status = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# ===================== DATA LOADING AND PREPROCESSING =====================

@st.cache_data
def load_raw_data():
    """Load the raw credit card dataset"""
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        return df
    except:
        return None

def preprocess_and_save():
    """Preprocess raw CSV and save cleaned CSV"""
    df = load_raw_data()
    if df is None:
        st.error("Raw dataset not found! Please ensure 'UCI_Credit_Card.csv' is in the same directory.")
        return None

    # Rename columns for better readability
    df.rename(columns={
        'LIMIT_BAL': 'credit_limit',
        'SEX': 'gender',
        'EDUCATION': 'education',
        'MARRIAGE': 'marriage',
        'AGE': 'age',
        'PAY_0': 'pay_sept',
        'PAY_2': 'pay_aug',
        'PAY_3': 'pay_july',
        'PAY_4': 'pay_june',
        'PAY_5': 'pay_may',
        'PAY_6': 'pay_april',
        'BILL_AMT1': 'bill_sept',
        'BILL_AMT2': 'bill_aug',
        'BILL_AMT3': 'bill_july',
        'BILL_AMT4': 'bill_june',
        'BILL_AMT5': 'bill_may',
        'BILL_AMT6': 'bill_april',
        'PAY_AMT1': 'payment_sept',
        'PAY_AMT2': 'payment_aug',
        'PAY_AMT3': 'payment_july',
        'PAY_AMT4': 'payment_june',
        'PAY_AMT5': 'payment_may',
        'PAY_AMT6': 'payment_april',
        'default.payment.next.month': 'default'
    }, inplace=True)

    # Drop ID column if exists
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

    # Fix categorical values
    df['education'] = df['education'].replace([0, 5, 6], 4)
    df['marriage'] = df['marriage'].replace(0, 3)

    # Create derived features
    df['total_bill'] = df[['bill_sept', 'bill_aug', 'bill_july', 'bill_june', 'bill_may', 'bill_april']].sum(axis=1)
    df['total_payment'] = df[['payment_sept', 'payment_aug', 'payment_july', 'payment_june', 'payment_may', 'payment_april']].sum(axis=1)
    df['payment_ratio'] = np.where(df['total_bill'] > 0, df['total_payment'] / df['total_bill'], 0)
    df['avg_delay'] = df[['pay_sept', 'pay_aug', 'pay_july', 'pay_june', 'pay_may', 'pay_april']].mean(axis=1)
    df['bill_payment_diff'] = df['total_bill'] - df['total_payment']
    df['payment_delay_severity'] = df[['pay_sept', 'pay_aug', 'pay_july', 'pay_june', 'pay_may', 'pay_april']].apply(lambda x: (x > 0).sum(), axis=1)

    # Save cleaned data
    df.to_csv(CLEAN_DATA_PATH, index=False)
    return df

def load_clean_data():
    if os.path.exists(CLEAN_DATA_PATH):
        return pd.read_csv(CLEAN_DATA_PATH)
    return None

def save_models(models_dict):
    """Save models using joblib with error handling"""
    try:
        joblib.dump(models_dict, MODEL_PATH)
        return True
    except Exception as e:
        st.error(f"Error saving models: {str(e)}")
        return False

def load_models():
    """Load models with error handling"""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None
    return None

@st.cache_resource
def create_preprocessor():
    """Create preprocessing pipeline"""
    categorical_features = ['gender', 'education', 'marriage']
    numerical_features = ['credit_limit', 'age', 'pay_sept', 'pay_aug', 'pay_july', 
                         'pay_june', 'pay_may', 'pay_april', 'bill_sept', 'bill_aug',
                         'bill_july', 'bill_june', 'bill_may', 'bill_april', 
                         'payment_sept', 'payment_aug', 'payment_july', 'payment_june',
                         'payment_may', 'payment_april', 'total_bill', 'total_payment',
                         'payment_ratio', 'avg_delay', 'bill_payment_diff', 'payment_delay_severity']
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numerical_features, categorical_features

def get_feature_names(preprocessor, numerical_features, categorical_features):
    """Get feature names after preprocessing"""
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)
    return all_feature_names

def train_models(_preprocessor, X_train, y_train, numerical_features, categorical_features):
    """Train Random Forest and XGBoost models with proper error handling"""
    
    models = {}
    training_status = {}
    
    # Fit preprocessor
    _preprocessor.fit(X_train)
    
    # Get feature names
    feature_names = get_feature_names(_preprocessor, numerical_features, categorical_features)
    
    # 1. Random Forest
    try:
        with st.spinner("Training Random Forest..."):
            rf_pipeline = Pipeline(steps=[
                ('preprocessor', _preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            rf_pipeline.fit(X_train, y_train)
            models['Random Forest'] = rf_pipeline
            training_status['Random Forest'] = '✅ Success'
    except Exception as e:
        training_status['Random Forest'] = f'❌ Failed: {str(e)[:50]}...'
        st.error(f"Random Forest training failed: {str(e)}")
    
    # 2. XGBoost
    try:
        with st.spinner("Training XGBoost..."):
            xgb_pipeline = Pipeline(steps=[
                ('preprocessor', _preprocessor),
                ('classifier', XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            xgb_pipeline.fit(X_train, y_train)
            models['XGBoost'] = xgb_pipeline
            training_status['XGBoost'] = '✅ Success'
    except Exception as e:
        training_status['XGBoost'] = f'❌ Failed: {str(e)[:50]}...'
        st.error(f"XGBoost training failed: {str(e)}")
    
    return models, training_status, feature_names

def evaluate_models(models_dict, X_test, y_test, threshold=0.3):
    """Evaluate Random Forest and XGBoost models and return metrics"""
    results = []
    y_pred_proba_dict = {}
    failed_models = []
    
    for name, pipeline in models_dict.items():
        try:
            # Handle sklearn pipelines
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            y_pred = (y_pred_proba >= threshold).astype(int)
            y_pred_proba_dict[name] = y_pred_proba
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
        except Exception as e:
            failed_models.append(name)
            st.warning(f"Could not evaluate {name}: {str(e)}")
    
    return pd.DataFrame(results), y_pred_proba_dict, failed_models

def get_feature_importance(model_info, feature_names):
    """Extract feature importance from model"""
    try:
        # Sklearn pipeline
        if hasattr(model_info.named_steps['classifier'], 'feature_importances_'):
            importance = model_info.named_steps['classifier'].feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
    except:
        return None

# ===================== SIDEBAR =====================

st.sidebar.image("logo.png")
st.sidebar.title("Credit Default Predictor")
st.sidebar.markdown("---")

# Navigation
tabs = [
    "🏠 Project Overview",
    "⚙️ Preprocess & Train",
    "📊 Data Exploration",
    "🔍 Feature Importance",
    "🤖 Model Comparison",
    "⚠️ Default Prediction",
    "📈 Risk Dashboard",
    "🔄 Scenario Simulator",
    "💡 Business Insights"
]

selected_tab = st.sidebar.radio("Navigate to:", tabs)
 
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application predicts credit card default using "
    "Random Forest and XGBoost algorithms. Built with Streamlit."
)

# ===================== MAIN CONTENT =====================

# Load data if available
if not st.session_state.data_loaded:
    df = load_clean_data()
    if df is not None:
        st.session_state.df = df
        st.session_state.data_loaded = True

# Load models if available
if not st.session_state.model_trained:
    models = load_models()
    if models is not None:
        st.session_state.models = models
        st.session_state.model_trained = True

# Main content based on selected tab
if selected_tab == "🏠 Project Overview":
    st.markdown("<h1 class='main-header'>Credit Card Default Prediction</h1>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Calculate dynamic metrics
        total_records = len(df)
        total_features = len(df.columns) - 1  # Subtract target column 'default'
        default_rate = df['default'].mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dataset Size", f"{total_records:,} records")
        with col2:
            st.metric("Features", f"{total_features} columns")
        with col3:
            st.metric("Models", f"{len(st.session_state.models)} ML Algorithms" if st.session_state.model_trained else "Not trained")
        with col4:
            st.metric("Default Rate", f"{default_rate:.2f}%")
        
        # Show additional dynamic info
        with st.expander("📊 Dataset Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Numerical Features:**")
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numerical_cols.remove('default') if 'default' in numerical_cols else None
                st.write(f"• {len(numerical_cols)} numerical features")
                
                st.markdown("**Categorical Features:**")
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.write(f"• {len(categorical_cols)} categorical features")
            
            with col2:  
                st.markdown("**Memory Usage:**")
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.write(f"• {memory_usage:.2f} MB")
                
                st.markdown("**Missing Values:**")
                missing_values = df.isnull().sum().sum()
                st.write(f"• {missing_values} missing values" if missing_values > 0 else "• No missing values")
    
    else:
        # Fallback when data is not loaded
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", "Not loaded")
        with col2:
            st.metric("Features", "N/A")
        with col3:
            st.metric("Models", "Not trained" if not st.session_state.model_trained else f"{len(st.session_state.models)} ML Algorithms")
        
        st.info("👈 Please go to **Preprocess & Train** tab to load the dataset.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Objective
        Predict whether a credit card client will default on their payment next month using machine learning.
        
        ### 📊 Dataset Overview
        - **Source**: UCI Machine Learning Repository
        - **Records**: 30,000 credit card clients in Taiwan
        - **Features**: 23 original features + 6 engineered features
        - **Target**: Default payment next month (Yes/No)
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Features
        - **Data Preprocessing**: Automated cleaning and feature engineering
        - **Two Powerful Models**: Random Forest and XGBoost
        - **Model Comparison**: Side-by-side performance metrics
        - **Feature Importance**: Understand key risk factors
        - **Interactive Predictions**: Test with custom inputs
        - **Risk Dashboard**: Visual analysis of risk factors
        - **Scenario Simulation**: What-if analysis
        """)
    
    if st.session_state.data_loaded:
        st.markdown("### 📈 Quick Stats")
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Default Rate", f"{df['default'].mean()*100:.2f}%")
        with col2:
            st.metric("Avg Credit Limit", f"${df['credit_limit'].mean():,.0f}")
        with col3:
            st.metric("Avg Age", f"{df['age'].mean():.1f} years")
        with col4:
            st.metric("Avg Payment Ratio", f"{df['payment_ratio'].mean():.2f}")

elif selected_tab == "⚙️ Preprocess & Train":
    st.markdown("<h2 class='sub-header'>⚙️ Preprocessing & Training Pipeline</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Data Preprocessing")
        if st.button("🔄 Preprocess Data", key="preprocess_btn"):
            with st.spinner("Preprocessing data..."):
                df = preprocess_and_save()
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success("✅ Data preprocessing completed and saved!")
                    st.balloons()
                else:
                    st.error("❌ Failed to preprocess data. Check if raw dataset exists.")
        
        if st.session_state.data_loaded:
            st.info(f"✅ Cleaned data loaded: {len(st.session_state.df)} records")
            if st.checkbox("Show cleaned data preview"):
                st.dataframe(st.session_state.df.head())
            
            # Show data info
            if st.checkbox("Show data info"):
                buffer = st.session_state.df.info(buf=None)
                st.text(f"Dataset shape: {st.session_state.df.shape}")
                st.text(f"Missing values: {st.session_state.df.isnull().sum().sum()}")
    
    with col2:
        st.markdown("### 🤖 Model Training")
        if st.button("🚀 Train Models", key="train_btn"):
            if st.session_state.data_loaded:
                with st.spinner("Training Random Forest and XGBoost models... This may take a few moments."):
                    df = st.session_state.df
                    
                    # Prepare data
                    X = df.drop('default', axis=1)
                    y = df['default']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Store data for later use
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Create preprocessor
                    preprocessor, num_features, cat_features = create_preprocessor()
                    st.session_state.preprocessor = preprocessor
                    
                    # Train models with status tracking
                    models, training_status, feature_names = train_models(
                        preprocessor, X_train, y_train, num_features, cat_features
                    )
                    st.session_state.feature_names = feature_names
                    
                    # Save models if any were trained successfully
                    if models:
                        if save_models(models):
                            # Update session state
                            st.session_state.models = models
                            st.session_state.model_trained = True
                            st.session_state.training_status = training_status
                            
                            # Show training status
                            st.markdown("### 📊 Training Status")
                            status_df = pd.DataFrame(list(training_status.items()), 
                                                    columns=['Model', 'Status'])
                            
                            # Color code the status
                            def color_status(val):
                                if '✅' in val:
                                    return 'background-color: #D1FAE5'
                                elif '❌' in val:
                                    return 'background-color: #FEE2E2'
                                return ''
                            
                            styled_df = status_df.style.map(color_status, subset=['Status'])
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Show success message with count
                            successful_models = sum(1 for status in training_status.values() if '✅' in status)
                            st.markdown(f"""
                            <div class='success-box'>
                                <h3>✅ Training Complete!</h3>
                                <p>{successful_models} out of {len(training_status)} models trained successfully.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show model performance for successful models
                            if successful_models > 0:
                                results_df, _, failed = evaluate_models(models, X_test, y_test)
                                if not results_df.empty:
                                    st.markdown("### 📊 Model Performance (Trained Models)")
                                    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
                    else:
                        st.error("❌ All models failed to train. Please check the error messages above.")
            else:
                st.error("❌ Please preprocess data first!")
        
        if st.session_state.model_trained and st.session_state.training_status:
            st.markdown("### Current Model Status")
            status_df = pd.DataFrame(list(st.session_state.training_status.items()), 
                                    columns=['Model', 'Status'])
            st.dataframe(status_df, use_container_width=True)

elif selected_tab == "📊 Data Exploration":
    st.markdown("<h2 class='sub-header'>📊 Data Exploration</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please preprocess data first in the 'Preprocess & Train' tab.")
    else:
        df = st.session_state.df
        
        # Default rate
        default_rate = df['default'].mean() * 100
        
        st.markdown("### 📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Default Rate", f"{default_rate:.2f}%", delta=f"{default_rate-22:.2f}%" if default_rate > 22 else f"{default_rate-22:.2f}%")
        with col2:
            st.metric("Total Records", f"{len(df):,}")
        with col3:
            st.metric("Features", len(df.columns))
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Target Distribution", "Numerical Features", "Categorical Features", "Correlation Analysis"])
        
        with viz_tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Pie chart
                fig = px.pie(values=df['default'].value_counts().values, 
                             names=['No Default (0)', 'Default (1)'],
                             title='Default Distribution',
                             color_discrete_sequence=['#10B981', '#DC2626'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart
                fig = go.Figure(data=[
                    go.Bar(name='Count', x=['No Default', 'Default'], 
                           y=df['default'].value_counts().values,
                           marker_color=['#10B981', '#DC2626'])
                ])
                fig.update_layout(title='Default Count', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            numerical_cols = ['credit_limit', 'age', 'payment_ratio', 'avg_delay', 
                            'total_bill', 'total_payment', 'bill_payment_diff']
            
            selected_num = st.multiselect("Select numerical features to visualize", 
                                         numerical_cols, default=['credit_limit', 'age', 'payment_ratio'])
            
            if selected_num:
                for feature in selected_num:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x=feature, color='default', 
                                          marginal='box', title=f'{feature} Distribution by Default',
                                          color_discrete_sequence=['#10B981', '#DC2626'])
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.box(df, x='default', y=feature, 
                                    title=f'{feature} by Default Status',
                                    color='default', color_discrete_sequence=['#10B981', '#DC2626'])
                        st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            categorical_cols = ['gender', 'education', 'marriage']
            
            for feature in categorical_cols:
                col1, col2 = st.columns(2)
                with col1:
                    # Count plot
                    cross_tab = pd.crosstab(df[feature], df['default'], normalize='index') * 100
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='No Default', x=cross_tab.index, y=cross_tab[0],
                                        marker_color='#10B981'))
                    fig.add_trace(go.Bar(name='Default', x=cross_tab.index, y=cross_tab[1],
                                        marker_color='#DC2626'))
                    fig.update_layout(title=f'{feature} - Default Rate %', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distribution
                    fig = px.pie(values=df[feature].value_counts().values,
                               names=df[feature].value_counts().index,
                               title=f'{feature} Distribution')
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab4:
            # Correlation matrix
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix, 
                           title='Feature Correlation Matrix',
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations with default
            default_corr = corr_matrix['default'].sort_values(ascending=False)
            st.markdown("### Top Features Correlated with Default")
            fig = px.bar(x=default_corr.index[1:11], y=default_corr.values[1:11],
                        title='Top 10 Features Correlated with Default',
                        color=default_corr.values[1:11],
                        color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "🔍 Feature Importance":
    st.markdown("<h2 class='sub-header'>🔍 Feature Importance Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the 'Preprocess & Train' tab.")
    else:
        models = st.session_state.models
        feature_names = st.session_state.feature_names
        
        if not feature_names:
            st.error("Feature names not available. Please retrain models.")
        else:
            # Model selection
            available_models = list(models.keys())
            
            selected_model = st.selectbox("Select Model for Feature Importance:", available_models)
            
            if selected_model:
                model_info = models[selected_model]
                
                # Get feature importance
                importance_df = None
                
                if hasattr(model_info.named_steps['classifier'], 'feature_importances_'):
                    importance = model_info.named_steps['classifier'].feature_importances_
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                
                if importance_df is not None:
                    st.markdown(f"### Top 20 Most Important Features - {selected_model}")
                    
                    # Plot top 20 features
                    top_20 = importance_df.head(20)
                    
                    fig = px.bar(top_20, x='importance', y='feature', 
                                orientation='h',
                                title=f'Feature Importance - {selected_model}',
                                color='importance',
                                color_continuous_scale='Viridis')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance table
                    st.markdown("### Feature Importance Table")
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # Cumulative importance
                    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
                    st.markdown("### Cumulative Importance")
                    fig = px.line(importance_df.head(50), y='cumulative_importance',
                                 title='Cumulative Feature Importance',
                                 markers=True)
                    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                 annotation_text="80% threshold")
                    fig.add_hline(y=0.9, line_dash="dash", line_color="green", 
                                 annotation_text="90% threshold")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature selection recommendation
                    st.markdown("### Feature Selection Recommendation")
                    features_for_80 = importance_df[importance_df['cumulative_importance'] <= 0.8].shape[0]
                    features_for_90 = importance_df[importance_df['cumulative_importance'] <= 0.9].shape[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🔹 {features_for_80} features explain 80% of importance")
                    with col2:
                        st.info(f"🔹 {features_for_90} features explain 90% of importance")
                else:
                    st.warning(f"Feature importance not available for {selected_model}")

elif selected_tab == "🤖 Model Comparison":
    st.markdown("<h2 class='sub-header'>🤖 Model Comparison</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the 'Preprocess & Train' tab.")
    else:
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        if not models:
            st.error("No trained models available. Please retrain in the 'Preprocess & Train' tab.")
        else:
            # Threshold slider
            threshold = st.slider("Select classification threshold:", 0.1, 0.9, 0.3, 0.05)
            
            # Evaluate models
            results_df, y_pred_proba_dict, failed_models = evaluate_models(models, X_test, y_test, threshold)
            
            if failed_models:
                st.warning(f"Could not evaluate: {', '.join(failed_models)}")
            
            if not results_df.empty:
                # Display metrics
                st.markdown("### Model Performance Metrics")
                
                # Color coding for metrics
                styled_results = results_df.style.highlight_max(axis=0, color='lightgreen')\
                                               .highlight_min(axis=0, color='lightcoral')
                st.dataframe(styled_results, use_container_width=True)
                
                # Radar chart for model comparison
                st.markdown("### Model Comparison Radar Chart")
                
                fig = go.Figure()
                for model in results_df['Model']:
                    model_data = results_df[results_df['Model'] == model].iloc[0]
                    fig.add_trace(go.Scatterpolar(
                        r=[model_data['Accuracy'], model_data['Precision'], 
                           model_data['Recall'], model_data['F1-Score'], model_data['ROC-AUC']],
                        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        fill='toself',
                        name=model
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curves
                st.markdown("### ROC Curves")
                fig = go.Figure()
                
                for name, y_pred_proba in y_pred_proba_dict.items():
                    if name not in failed_models:
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        auc = roc_auc_score(y_test, y_pred_proba)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                                name=f'{name} (AUC={auc:.3f})'))
                
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                        name='Random', line=dict(dash='dash', color='gray')))
                fig.update_layout(title='ROC Curves Comparison',
                                 xaxis_title='False Positive Rate',
                                 yaxis_title='True Positive Rate')
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar chart comparison
                st.markdown("### Metrics Comparison Bar Chart")
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                
                for model in results_df['Model']:
                    model_data = results_df[results_df['Model'] == model].iloc[0]
                    fig.add_trace(go.Bar(
                        name=model,
                        x=metrics,
                        y=[model_data[m] for m in metrics]
                    ))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    xaxis_title='Metrics',
                    yaxis_title='Score',
                    barmode='group',
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "⚠️ Default Prediction":
    st.markdown("<h2 class='sub-header'>⚠️ Default Prediction</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the 'Preprocess & Train' tab.")
    elif not st.session_state.models:
        st.error("No trained models available. Please retrain in the 'Preprocess & Train' tab.")
    else:
        models = st.session_state.models
        
        st.markdown("### Enter Customer Details")
        
        with st.expander("Customer Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                credit_limit = st.number_input("Credit Limit ($)", min_value=10000, max_value=1000000, value=50000, step=10000)
                gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
                education = st.selectbox("Education", options=[1, 2, 3, 4], 
                                        format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])
                marriage = st.selectbox("Marital Status", options=[1, 2, 3],
                                       format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
                age = st.number_input("Age", min_value=21, max_value=79, value=35)
            
            with col2:
                st.markdown("#### Payment History (Last 6 months)")
                pay_sept = st.selectbox("September Payment Status", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_aug = st.selectbox("August Payment Status", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_july = st.selectbox("July Payment Status", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_june = st.selectbox("June Payment Status", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_may = st.selectbox("May Payment Status", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_april = st.selectbox("April Payment Status", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            
            with col3:
                st.markdown("#### Bill Amounts ($)")
                bill_sept = st.number_input("September Bill", value=10000)
                bill_aug = st.number_input("August Bill", value=10000)
                bill_july = st.number_input("July Bill", value=10000)
                bill_june = st.number_input("June Bill", value=10000)
                bill_may = st.number_input("May Bill", value=10000)
                bill_april = st.number_input("April Bill", value=10000)
        
        with st.expander("Payment Amounts", expanded=True):
            col4, col5, col6 = st.columns(3)
            with col4:
                payment_sept = st.number_input("September Payment ($)", value=5000)
                payment_aug = st.number_input("August Payment ($)", value=5000)
            with col5:
                payment_july = st.number_input("July Payment ($)", value=5000)
                payment_june = st.number_input("June Payment ($)", value=5000)
            with col6:
                payment_may = st.number_input("May Payment ($)", value=5000)
                payment_april = st.number_input("April Payment ($)", value=5000)
        
        if st.button("🔮 Predict Default Probability", type="primary"):
            # Calculate derived features
            total_bill = bill_sept + bill_aug + bill_july + bill_june + bill_may + bill_april
            total_payment = payment_sept + payment_aug + payment_july + payment_june + payment_may + payment_april
            payment_ratio = total_payment / total_bill if total_bill > 0 else 0
            avg_delay = np.mean([pay_sept, pay_aug, pay_july, pay_june, pay_may, pay_april])
            bill_payment_diff = total_bill - total_payment
            payment_delay_severity = sum([1 for p in [pay_sept, pay_aug, pay_july, pay_june, pay_may, pay_april] if p > 0])
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'credit_limit': [credit_limit],
                'gender': [gender],
                'education': [education],
                'marriage': [marriage],
                'age': [age],
                'pay_sept': [pay_sept],
                'pay_aug': [pay_aug],
                'pay_july': [pay_july],
                'pay_june': [pay_june],
                'pay_may': [pay_may],
                'pay_april': [pay_april],
                'bill_sept': [bill_sept],
                'bill_aug': [bill_aug],
                'bill_july': [bill_july],
                'bill_june': [bill_june],
                'bill_may': [bill_may],
                'bill_april': [bill_april],
                'payment_sept': [payment_sept],
                'payment_aug': [payment_aug],
                'payment_july': [payment_july],
                'payment_june': [payment_june],
                'payment_may': [payment_may],
                'payment_april': [payment_april],
                'total_bill': [total_bill],
                'total_payment': [total_payment],
                'payment_ratio': [payment_ratio],
                'avg_delay': [avg_delay],
                'bill_payment_diff': [bill_payment_diff],
                'payment_delay_severity': [payment_delay_severity]
            })
            
            # Make predictions with each model
            st.markdown("### Prediction Results")
            
            results = []
            valid_predictions = []
            
            for name, pipeline in models.items():
                try:
                    proba = pipeline.predict_proba(input_data)[0][1]
                    
                    results.append({
                        'Model': name,
                        'Default Probability': f"{proba:.2%}",
                        'Prediction': '⚠️ Default' if proba >= 0.5 else '✅ No Default'
                    })
                    valid_predictions.append(proba)
                except Exception as e:
                    results.append({
                        'Model': name,
                        'Default Probability': 'Error',
                        'Prediction': f'❌ Failed'
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Ensemble prediction
                if valid_predictions:
                    avg_proba = np.mean(valid_predictions)
                    std_proba = np.std(valid_predictions)
                    
                    st.markdown("### Ensemble Prediction")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Probability", f"{avg_proba:.2%}")
                    with col2:
                        risk_level = "High" if avg_proba >= 0.5 else "Medium" if avg_proba >= 0.3 else "Low"
                        risk_class = "risk-high" if avg_proba >= 0.5 else "risk-medium" if avg_proba >= 0.3 else "risk-low"
                        st.markdown(f"<div class='{risk_class}'>Risk Level: {risk_level}</div>", unsafe_allow_html=True)
                    with col3:
                        confidence = max(avg_proba, 1-avg_proba)
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col4:
                        st.metric("Model Agreement", f"{(1 - std_proba/0.5):.2%}")

elif selected_tab == "📈 Risk Dashboard":
    st.markdown("<h2 class='sub-header'>📈 Risk Dashboard</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please preprocess data first in the 'Preprocess & Train' tab.")
    else:
        df = st.session_state.df
        
        # Risk metrics
        st.markdown("### 📊 Key Risk Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Default Rate", f"{df['default'].mean()*100:.2f}%")
        with col2:
            high_risk = df[df['payment_ratio'] < 0.3]['default'].mean() * 100
            st.metric("High Risk Group Default", f"{high_risk:.2f}%", delta=f"{high_risk - df['default'].mean()*100:.1f}%")
        with col3:
            low_risk = df[df['payment_ratio'] > 0.8]['default'].mean() * 100
            st.metric("Low Risk Group Default", f"{low_risk:.2f}%", delta=f"{low_risk - df['default'].mean()*100:.1f}%")
        with col4:
            st.metric("Avg Payment Ratio", f"{df['payment_ratio'].mean():.2f}")
        
        # Risk by segments
        st.markdown("### 🎯 Risk by Segments")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Payment Behavior", "Credit Utilization", "Risk Profiles"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Risk by age group
                df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 80], 
                                        labels=['20-30', '30-40', '40-50', '50-60', '60+'])
                age_risk = df.groupby('age_group', observed=True)['default'].agg(['mean', 'count']).reset_index()
                age_risk['mean'] = age_risk['mean'] * 100
                
                fig = px.bar(age_risk, x='age_group', y='mean', 
                            title='Default Rate by Age Group',
                            text=age_risk['count'],
                            color='mean', color_continuous_scale='RdYlGn_r')
                fig.update_traces(texttemplate='%{text} customers', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk by education
                edu_risk = df.groupby('education')['default'].agg(['mean', 'count']).reset_index()
                edu_risk['education'] = edu_risk['education'].map({1: 'Grad School', 2: 'University', 
                                                                  3: 'High School', 4: 'Others'})
                edu_risk['mean'] = edu_risk['mean'] * 100
                
                fig = px.bar(edu_risk, x='education', y='mean',
                            title='Default Rate by Education',
                            text=edu_risk['count'],
                            color='mean', color_continuous_scale='RdYlGn_r')
                fig.update_traces(texttemplate='%{text} customers', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                # Risk by payment delay
                df['delay_category'] = pd.cut(df['avg_delay'], 
                                              bins=[-3, -1, 0, 2, 5, 10],
                                              labels=['Early', 'On Time', '1-2 months', '3-5 months', '6+ months'])
                delay_risk = df.groupby('delay_category', observed=True)['default'].mean() * 100
                
                fig = px.bar(x=delay_risk.index, y=delay_risk.values,
                            title='Default Rate by Payment Delay',
                            color=delay_risk.values,
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk by payment ratio
                df['payment_ratio_category'] = pd.cut(df['payment_ratio'],
                                                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                                      labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
                ratio_risk = df.groupby('payment_ratio_category', observed=True)['default'].mean() * 100
                
                fig = px.line(x=ratio_risk.index, y=ratio_risk.values,
                             title='Default Rate by Payment Ratio',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                # Risk by credit limit
                df['credit_limit_group'] = pd.qcut(df['credit_limit'], q=5,
                                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                credit_risk = df.groupby('credit_limit_group', observed=True)['default'].mean() * 100
                
                fig = px.bar(x=credit_risk.index, y=credit_risk.values,
                            title='Default Rate by Credit Limit',
                            color=credit_risk.values,
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk by bill amount
                df['bill_group'] = pd.qcut(df['total_bill'], q=5,
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                bill_risk = df.groupby('bill_group', observed=True)['default'].mean() * 100
                
                fig = px.bar(x=bill_risk.index, y=bill_risk.values,
                            title='Default Rate by Total Bill',
                            color=bill_risk.values,
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Create risk score
            df['risk_score'] = (df['avg_delay'] * 10 + (1 - df['payment_ratio']) * 100).clip(0, 100)
            
            col1, col2 = st.columns(2)
            with col1:
                # Risk score distribution
                fig = px.histogram(df, x='risk_score', color='default',
                                  title='Risk Score Distribution by Default',
                                  color_discrete_sequence=['#10B981', '#DC2626'],
                                  nbins=50)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk score by default
                fig = px.box(df, x='default', y='risk_score',
                            title='Risk Score by Default Status',
                            color='default', color_discrete_sequence=['#10B981', '#DC2626'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk matrix
            st.markdown("### Risk Matrix")
            df['risk_category'] = pd.cut(df['risk_score'],
                                         bins=[0, 30, 60, 100],
                                         labels=['Low Risk', 'Medium Risk', 'High Risk'])
            risk_matrix = pd.crosstab(df['risk_category'], df['default'], normalize='index') * 100
            
            fig = px.imshow(risk_matrix.values,
                           x=['No Default', 'Default'],
                           y=['Low Risk', 'Medium Risk', 'High Risk'],
                           text_auto='.1f',
                           color_continuous_scale='RdYlGn',
                           title='Risk Matrix - Default Probability by Risk Category')
            st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "🔄 Scenario Simulator":
    st.markdown("<h2 class='sub-header'>🔄 Scenario Simulator</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the 'Preprocess & Train' tab.")
    else:
        models = st.session_state.models
        
        st.markdown("### Simulate Different Customer Scenarios")
        st.markdown("Adjust the parameters below to see how default probability changes:")
        
        # Base scenario
        st.markdown("#### Base Scenario")
        col1, col2, col3 = st.columns(3)
        with col1:
            base_credit = st.number_input("Base Credit Limit", value=50000, step=10000)
            base_age = st.number_input("Base Age", value=35, step=5)
        with col2:
            base_payment_ratio = st.slider("Base Payment Ratio", 0.0, 1.0, 0.5, 0.1)
            base_avg_delay = st.slider("Base Average Delay", -2.0, 8.0, 0.0, 0.5)
        with col3:
            base_gender = st.selectbox("Base Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            base_education = st.selectbox("Base Education", [1, 2, 3, 4])
        
        if st.button("Run Scenario Analysis"):
            # Create variations
            scenarios = {
                'Base': (base_credit, base_payment_ratio, base_avg_delay),
                'High Credit': (base_credit * 2, base_payment_ratio, base_avg_delay),
                'Low Credit': (base_credit * 0.5, base_payment_ratio, base_avg_delay),
                'Good Payment': (base_credit, base_payment_ratio * 1.5, base_avg_delay * 0.5),
                'Poor Payment': (base_credit, base_payment_ratio * 0.5, base_avg_delay * 2),
                'Old Age': (base_credit, base_payment_ratio, base_avg_delay, base_age + 20),
                'Young Age': (base_credit, base_payment_ratio, base_avg_delay, base_age - 10)
            }
            
            results = []
            for scenario_name, params in scenarios.items():
                # Create input data for scenario
                input_data = pd.DataFrame({
                    'credit_limit': [params[0]],
                    'gender': [base_gender],
                    'education': [base_education],
                    'marriage': [1],
                    'age': [params[3] if len(params) > 3 else base_age],
                    'pay_sept': [params[2] * 2],
                    'pay_aug': [params[2] * 1.8],
                    'pay_july': [params[2] * 1.5],
                    'pay_june': [params[2] * 1.2],
                    'pay_may': [params[2]],
                    'pay_april': [params[2] * 0.8],
                    'bill_sept': [50000],
                    'bill_aug': [48000],
                    'bill_july': [45000],
                    'bill_june': [42000],
                    'bill_may': [40000],
                    'bill_april': [38000],
                    'payment_sept': [50000 * params[1]],
                    'payment_aug': [48000 * params[1]],
                    'payment_july': [45000 * params[1]],
                    'payment_june': [42000 * params[1]],
                    'payment_may': [40000 * params[1]],
                    'payment_april': [38000 * params[1]],
                    'total_bill': [50000 + 48000 + 45000 + 42000 + 40000 + 38000],
                    'total_payment': [(50000 + 48000 + 45000 + 42000 + 40000 + 38000) * params[1]],
                    'payment_ratio': [params[1]],
                    'avg_delay': [params[2]],
                    'bill_payment_diff': [(50000 + 48000 + 45000 + 42000 + 40000 + 38000) * (1 - params[1])],
                    'payment_delay_severity': [3 if params[2] > 0 else 0]
                })
                
                # Get predictions
                probs = []
                for name, pipeline in models.items():
                    try:
                        prob = pipeline.predict_proba(input_data)[0][1]
                        probs.append(prob)
                    except:
                        pass
                
                if probs:
                    avg_prob = np.mean(probs)
                    results.append({
                        'Scenario': scenario_name,
                        'Default Probability': f"{avg_prob:.2%}",
                        'Risk Level': 'High' if avg_prob >= 0.5 else 'Medium' if avg_prob >= 0.3 else 'Low'
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                st.markdown("### Scenario Analysis Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Visualize
                fig = px.bar(results_df, x='Scenario', y='Default Probability',
                            title='Default Probability by Scenario',
                            color='Risk Level',
                            color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#DC2626'})
                st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "💡 Business Insights":
    st.markdown("<h2 class='sub-header'>💡 Business Insights & Recommendations</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please preprocess data first in the 'Preprocess & Train' tab.")
    else:
        df = st.session_state.df
        
        st.markdown("""
        ### Key Business Insights
        
        Based on the analysis of credit card default data using Random Forest and XGBoost, here are the key insights and recommendations:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 Customer Segmentation
            
            **High Risk Segments:**
            - Customers with payment delays > 2 months
            - Low payment ratio (< 30%)
            - Young customers (age < 30) with high credit limits
            - Single individuals with low education levels
            
            **Low Risk Segments:**
            - Customers with consistent on-time payments
            - High payment ratio (> 80%)
            - Middle-aged customers (40-50 years)
            - Married individuals with graduate education
            """)
            
            st.markdown("""
            #### 🎯 Risk Management Strategies
            
            1. **Early Warning System**
               - Monitor payment delays monthly
               - Flag customers with 2+ consecutive delays
               - Send reminders before due dates
            
            2. **Credit Limit Optimization**
               - Reduce limits for high-risk segments
               - Increase limits for loyal, low-risk customers
               - Implement dynamic limit adjustments
            """)
        
        with col2:
            st.markdown("""
            #### 💰 Revenue Optimization
            
            1. **Targeted Marketing**
               - Offer balance transfer deals to at-risk customers
               - Provide loyalty rewards to low-risk segments
               - Cross-sell products based on risk profile
            
            2. **Collection Strategies**
               - Early intervention for high-risk accounts
               - Automated reminders for medium-risk
               - Personalized collection calls for severe cases
            
            #### 📈 Key Performance Indicators to Track
            
            - Default rate by segment
            - Collection recovery rate
            - Customer lifetime value by risk group
            - ROI of risk mitigation strategies
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Risk-Based Customer Segmentation")
        
        # Create segments
        df['segment'] = 'Medium Risk'
        df.loc[(df['payment_ratio'] > 0.8) & (df['avg_delay'] < 0), 'segment'] = 'Low Risk'
        df.loc[(df['payment_ratio'] < 0.3) | (df['avg_delay'] > 2), 'segment'] = 'High Risk'
        
        segment_stats = df.groupby('segment').agg({
            'default': 'mean',
            'credit_limit': 'mean',
            'age': 'mean'
        }).reset_index()
        segment_stats['default'] = segment_stats['default'] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(segment_stats, values='default', names='segment',
                        title='Default Rate by Segment',
                        color='segment',
                        color_discrete_map={'Low Risk': '#10B981', 
                                          'Medium Risk': '#F59E0B', 
                                          'High Risk': '#DC2626'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_counts = df['segment'].value_counts().reset_index()
            segment_counts.columns = ['segment', 'count']
            fig = px.bar(segment_counts, x='segment', y='count',
                        title='Customer Distribution by Segment',
                        color='segment',
                        color_discrete_map={'Low Risk': '#10B981', 
                                          'Medium Risk': '#F59E0B', 
                                          'High Risk': '#DC2626'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 📝 Recommendations Summary
        
        1. **Immediate Actions:**
           - Implement real-time payment monitoring
           - Set up automated alerts for delayed payments
           - Review credit limits for high-risk customers
        
        2. **Short-term Initiatives (3-6 months):**
           - Develop segmented collection strategies
           - Launch targeted retention campaigns
           - Enhance credit assessment process
        
        3. **Long-term Strategy (6-12 months):**
           - Build predictive models for early warning
           - Implement dynamic pricing based on risk
           - Develop customer education programs
        
        ### 💡 Expected Impact
        
        - **15-20% reduction** in default rates
        - **25% improvement** in collection efficiency
        - **10-15% increase** in customer retention
        - **$2-3M annual savings** for a portfolio of 100K customers
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        Credit Card Default Prediction App | Built with Random Forest & XGBoost | Streamlit, Python, and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)