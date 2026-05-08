"""
Premier League Betting Strategy - COMPLETE ROLLING MODIFICATION MODEL (FINAL)
Data source: GitHub raw URL
Output: Display and save charts (No IRR, No ROI vs Accuracy)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ========== 设置保存路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else os.getcwd()
SAVE_PATH = SCRIPT_DIR
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Files will be saved to: {SAVE_PATH}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def log(msg):
    print(msg)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """Load and preprocess historical data from GitHub"""
    log("="*80)
    log("STEP 1: LOADING HISTORICAL DATA FROM GITHUB")
    log("="*80)
    
    url =  "https://raw.githubusercontent.com/loyal151009-star/EPL-Betting-Strategy/main/Data_EPL%20Season22-26.csv"
    log(f"Downloading from: {url}")
    
    df = pd.read_csv(url)
    log(f"  Successfully loaded {len(df)} rows")
    
    df = pd.read_csv(url)
    log(f"  Successfully loaded {len(df)} rows")
    
    df_clean = df[['DATE', 'HOMETEAM', 'AWAYTEAM', 'RESULT', 'BET365H', 'BET365D', 'BET365A']].copy()
    df_clean['DATE'] = pd.to_datetime(df_clean['DATE'], format='%d/%m/%Y')
    df_clean['MONTH'] = df_clean['DATE'].dt.month
    df_clean['YEAR'] = df_clean['DATE'].dt.year
    
    result_map = {'H': 2, 'D': 1, 'A': 0}
    df_clean['RESULT_NUM'] = df_clean['RESULT'].map(result_map)
    df_clean = df_clean.dropna(subset=['BET365H', 'BET365D', 'BET365A'])
    
    # Time decay weighting (only for historical matches with results)
    historical_mask = df_clean['RESULT_NUM'].notna()
    if historical_mask.any():
        min_date = df_clean.loc[historical_mask, 'DATE'].min()
        df_clean.loc[historical_mask, 'DAYS_FROM_START'] = (df_clean.loc[historical_mask, 'DATE'] - min_date).dt.days
        max_days = df_clean.loc[historical_mask, 'DAYS_FROM_START'].max()
        df_clean.loc[historical_mask, 'TIME_WEIGHT'] = np.exp(2 * df_clean.loc[historical_mask, 'DAYS_FROM_START'] / max_days) - 0.5
        df_clean.loc[historical_mask, 'TIME_WEIGHT'] = df_clean.loc[historical_mask, 'TIME_WEIGHT'] / df_clean.loc[historical_mask, 'TIME_WEIGHT'].max()
    
    df_clean.loc[~historical_mask, 'TIME_WEIGHT'] = 1.0
    df_clean.loc[~historical_mask, 'DAYS_FROM_START'] = 0
    
    # Season stage
    def get_season_stage(month):
        if month in [8, 9]: return 0
        elif month in [10, 11, 12]: return 1
        elif month in [1, 2]: return 2
        else: return 3
    
    df_clean['SEASON_STAGE'] = df_clean['MONTH'].apply(get_season_stage)
    
    historical_count = historical_mask.sum()
    upcoming_count = (~historical_mask).sum()
    
    log(f"  Loaded {len(df_clean)} matches total")
    log(f"  Historical matches (with results): {historical_count}")
    log(f"  Upcoming matches (to predict): {upcoming_count}")
    
    if historical_count > 0:
        result_dist = df_clean.loc[historical_mask, 'RESULT'].value_counts()
        log(f"  Historical results: H={result_dist.get('H',0)} ({result_dist.get('H',0)/historical_count*100:.1f}%), "
            f"D={result_dist.get('D',0)} ({result_dist.get('D',0)/historical_count*100:.1f}%), "
            f"A={result_dist.get('A',0)} ({result_dist.get('A',0)/historical_count*100:.1f}%)")
    
    return df_clean

def create_team_encoders(df):
    """Create team label encoders using all teams"""
    all_teams = pd.concat([df['HOMETEAM'], df['AWAYTEAM']]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)
    log(f"  Unique teams: {len(all_teams)}")
    return team_encoder, all_teams

def calculate_team_form(df, team, current_date, n_games=5):
    """Calculate team form over last N games"""
    historical_df = df[df['RESULT_NUM'].notna()]
    team_games = historical_df[((historical_df['HOMETEAM'] == team) | (historical_df['AWAYTEAM'] == team)) & 
                                (historical_df['DATE'] < current_date)].sort_values('DATE', ascending=False).head(n_games)
    
    if len(team_games) == 0:
        return 0.35, 0.25, 0.5
    
    wins = 0
    draws = 0
    points = 0
    
    for _, row in team_games.iterrows():
        if row['HOMETEAM'] == team and row['RESULT'] == 'H':
            wins += 1
            points += 3
        elif row['AWAYTEAM'] == team and row['RESULT'] == 'A':
            wins += 1
            points += 3
        elif row['RESULT'] == 'D':
            draws += 1
            points += 1
    
    momentum = points / (len(team_games) * 3)
    
    return wins / len(team_games), draws / len(team_games), momentum

def calculate_h2h(df, home_team, away_team, current_date):
    """Calculate head-to-head record"""
    historical_df = df[df['RESULT_NUM'].notna()]
    prev_meetings = historical_df[((historical_df['HOMETEAM'] == home_team) & (historical_df['AWAYTEAM'] == away_team)) |
                                   ((historical_df['HOMETEAM'] == away_team) & (historical_df['AWAYTEAM'] == home_team))]
    prev_meetings = prev_meetings[prev_meetings['DATE'] < current_date]
    
    if len(prev_meetings) == 0:
        return 0.35
    
    home_wins = 0
    for _, row in prev_meetings.iterrows():
        if row['HOMETEAM'] == home_team and row['RESULT'] == 'H':
            home_wins += 1
        elif row['AWAYTEAM'] == home_team and row['RESULT'] == 'A':
            home_wins += 1
    
    return home_wins / len(prev_meetings)

def create_features(df, team_encoder, all_teams):
    """Create all features for training"""
    log("\n" + "="*80)
    log("STEP 2: FEATURE ENGINEERING")
    log("="*80)
    df = df.copy()
    df = df.sort_values('DATE').reset_index(drop=True)
    
    df['HOME_TEAM_ENCODED'] = team_encoder.transform(df['HOMETEAM'])
    df['AWAY_TEAM_ENCODED'] = team_encoder.transform(df['AWAYTEAM'])
    
    df['HOME_LAST5_WIN_RATE'] = 0.35
    df['HOME_LAST5_DRAW_RATE'] = 0.25
    df['HOME_MOMENTUM'] = 0.5
    df['AWAY_LAST5_WIN_RATE'] = 0.35
    df['AWAY_LAST5_DRAW_RATE'] = 0.25
    df['AWAY_MOMENTUM'] = 0.5
    df['HEAD_TO_HEAD_HOME_WIN_RATE'] = 0.35
    
    log("  Calculating team form features...")
    for idx in range(len(df)):
        if idx % 500 == 0 and idx > 0:
            log(f"    Processing {idx}/{len(df)}...")
        
        current_date = df.loc[idx, 'DATE']
        home_team = df.loc[idx, 'HOMETEAM']
        away_team = df.loc[idx, 'AWAYTEAM']
        
        home_win, home_draw, home_mom = calculate_team_form(df, home_team, current_date, 5)
        df.loc[idx, 'HOME_LAST5_WIN_RATE'] = home_win
        df.loc[idx, 'HOME_LAST5_DRAW_RATE'] = home_draw
        df.loc[idx, 'HOME_MOMENTUM'] = home_mom
        
        away_win, away_draw, away_mom = calculate_team_form(df, away_team, current_date, 5)
        df.loc[idx, 'AWAY_LAST5_WIN_RATE'] = away_win
        df.loc[idx, 'AWAY_LAST5_DRAW_RATE'] = away_draw
        df.loc[idx, 'AWAY_MOMENTUM'] = away_mom
        
        h2h = calculate_h2h(df, home_team, away_team, current_date)
        df.loc[idx, 'HEAD_TO_HEAD_HOME_WIN_RATE'] = h2h
    
    # Odds-derived features
    df['HOME_IMPLIED'] = 1 / df['BET365H']
    df['DRAW_IMPLIED'] = 1 / df['BET365D']
    df['AWAY_IMPLIED'] = 1 / df['BET365A']
    
    total_implied = df['HOME_IMPLIED'] + df['DRAW_IMPLIED'] + df['AWAY_IMPLIED']
    
    df['HOME_NORM_PROB'] = df['HOME_IMPLIED'] / total_implied
    df['DRAW_NORM_PROB'] = df['DRAW_IMPLIED'] / total_implied
    df['AWAY_NORM_PROB'] = df['AWAY_IMPLIED'] / total_implied
    
    df['HOME_VALUE'] = df['HOME_NORM_PROB'] * df['BET365H']
    df['DRAW_VALUE'] = df['DRAW_NORM_PROB'] * df['BET365D']
    df['AWAY_VALUE'] = df['AWAY_NORM_PROB'] * df['BET365A']
    
    df['BOOKMAKER_OVERROUND'] = total_implied - 1
    df['ODDS_SPREAD'] = df[['BET365H', 'BET365D', 'BET365A']].max(axis=1) - df[['BET365H', 'BET365D', 'BET365A']].min(axis=1)
    df['HOME_ADVANTAGE'] = df['HOME_NORM_PROB'] - df['AWAY_NORM_PROB']
    
    log(f"  Features created successfully")
    
    return df

# ============================================================================
# 2. MODEL ARCHITECTURE DIAGRAMS
# ============================================================================

def plot_and_save(fig, filename):
    """Helper function to display and save a figure"""
    filepath = os.path.join(SAVE_PATH, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.5)
    log(f"✓ Saved and displayed: {filepath}")

def plot_random_forest_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Random Forest Architecture', fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.90, 'Ensemble of Bagged Decision Trees', fontsize=12, ha='center', transform=ax.transAxes, style='italic')
    ax.add_patch(plt.Rectangle((0.35, 0.75), 0.30, 0.08, facecolor='#3498db', edgecolor='black'))
    ax.text(0.5, 0.79, 'Training Data (n samples, m features)', fontsize=10, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    for i in range(3):
        x_start = 0.35 + i * 0.15
        ax.annotate('', xy=(x_start + 0.05, 0.68), xytext=(x_start + 0.05, 0.75), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.text(x_start + 0.05, 0.71, f'Bootstrap\nSample {i+1}', fontsize=7, ha='center', transform=ax.transAxes)
    for i in range(3):
        x_center = 0.35 + i * 0.15 + 0.05
        ax.add_patch(plt.Rectangle((x_center - 0.06, 0.55), 0.12, 0.10, facecolor='#2ecc71', edgecolor='black'))
        ax.text(x_center, 0.60, f'Tree {i+1}', fontsize=8, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        ax.text(x_center, 0.57, f'(max_depth=10)', fontsize=6, ha='center', va='center', transform=ax.transAxes)
    for i in range(3):
        x_center = 0.35 + i * 0.15 + 0.05
        ax.annotate('', xy=(x_center, 0.48), xytext=(x_center, 0.55), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((0.35, 0.38), 0.30, 0.10, facecolor='#e74c3c', edgecolor='black'))
    ax.text(0.5, 0.43, 'Majority Voting / Averaging', fontsize=10, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.annotate('', xy=(0.5, 0.30), xytext=(0.5, 0.38), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.add_patch(plt.Rectangle((0.40, 0.22), 0.20, 0.08, facecolor='#9b59b6', edgecolor='black'))
    ax.text(0.5, 0.26, 'Final Prediction', fontsize=10, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.add_patch(plt.Rectangle((0.05, 0.22), 0.25, 0.25, facecolor='#ecf0f1', edgecolor='black', alpha=0.7))
    ax.text(0.175, 0.44, 'Key Features:', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.175, 0.39, '• Bagging', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.175, 0.35, '• Feature randomness', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.175, 0.31, '• OOB evaluation', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.175, 0.27, '• Feature importance', fontsize=7, ha='center', transform=ax.transAxes)
    plot_and_save(fig, 'A1_random_forest_architecture.png')

def plot_gradient_boosting_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Gradient Boosting Architecture', fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.90, 'Sequential Ensemble with Gradient Descent', fontsize=12, ha='center', transform=ax.transAxes, style='italic')
    ax.add_patch(plt.Rectangle((0.35, 0.80), 0.30, 0.07, facecolor='#3498db', edgecolor='black'))
    ax.text(0.5, 0.835, 'Initial Prediction (Base Model)', fontsize=9, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.annotate('', xy=(0.5, 0.73), xytext=(0.5, 0.80), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    y_positions = [0.66, 0.59, 0.52, 0.45]
    for i, y in enumerate(y_positions):
        ax.add_patch(plt.Rectangle((0.35, y), 0.30, 0.06, facecolor='#2ecc71', edgecolor='black'))
        ax.text(0.5, y + 0.03, f'Tree {i+1} (Corrects previous errors)', fontsize=8, ha='center', va='center', transform=ax.transAxes)
        if i < len(y_positions) - 1:
            ax.annotate('', xy=(0.5, y - 0.01), xytext=(0.5, y + 0.06), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            ax.text(0.65, y + 0.025, f'LR η=0.05', fontsize=6, ha='center', transform=ax.transAxes)
    ax.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.45), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((0.35, 0.30), 0.30, 0.08, facecolor='#e74c3c', edgecolor='black'))
    ax.text(0.5, 0.34, 'Final Ensemble Prediction', fontsize=10, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.add_patch(plt.Rectangle((0.70, 0.30), 0.25, 0.25, facecolor='#ecf0f1', edgecolor='black', alpha=0.7))
    ax.text(0.825, 0.52, 'Key Features:', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.47, '• Sequential training', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.43, '• Gradient descent', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.39, '• Learning rate', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.35, '• Subsampling', fontsize=7, ha='center', transform=ax.transAxes)
    plot_and_save(fig, 'A2_gradient_boosting_architecture.png')

def plot_xgboost_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.text(0.5, 0.95, 'XGBoost Architecture', fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.90, 'Extreme Gradient Boosting with Regularization', fontsize=12, ha='center', transform=ax.transAxes, style='italic')
    ax.add_patch(plt.Rectangle((0.35, 0.80), 0.30, 0.07, facecolor='#3498db', edgecolor='black'))
    ax.text(0.5, 0.835, 'Training Data + Weights', fontsize=9, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.annotate('', xy=(0.5, 0.73), xytext=(0.5, 0.80), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((0.32, 0.65), 0.36, 0.08, facecolor='#e67e22', edgecolor='black'))
    ax.text(0.5, 0.69, 'Objective: Loss + Regularization', fontsize=9, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.text(0.5, 0.665, 'Ω(f) = γT + ½λ||w||²', fontsize=7, ha='center', va='center', transform=ax.transAxes, color='white')
    ax.annotate('', xy=(0.5, 0.57), xytext=(0.5, 0.65), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((0.32, 0.48), 0.36, 0.09, facecolor='#2ecc71', edgecolor='black'))
    ax.text(0.5, 0.525, 'Greedy Tree Building', fontsize=9, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.text(0.5, 0.495, 'Approximate Quantile Sketching', fontsize=7, ha='center', va='center', transform=ax.transAxes)
    ax.annotate('', xy=(0.5, 0.40), xytext=(0.5, 0.48), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((0.32, 0.32), 0.36, 0.08, facecolor='#1abc9c', edgecolor='black'))
    ax.text(0.5, 0.36, 'Parallel & Distributed Computing', fontsize=9, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.annotate('', xy=(0.5, 0.24), xytext=(0.5, 0.32), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.add_patch(plt.Rectangle((0.35, 0.16), 0.30, 0.08, facecolor='#e74c3c', edgecolor='black'))
    ax.text(0.5, 0.20, 'Final Ensemble Prediction', fontsize=10, ha='center', va='center', transform=ax.transAxes, color='white', fontweight='bold')
    ax.add_patch(plt.Rectangle((0.70, 0.40), 0.25, 0.30, facecolor='#ecf0f1', edgecolor='black', alpha=0.7))
    ax.text(0.825, 0.67, 'Key Features:', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.62, '• L1/L2 Regularization', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.58, '• Tree Pruning', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.54, '• Column Subsampling', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.50, '• Missing Value Handling', fontsize=7, ha='center', transform=ax.transAxes)
    ax.text(0.825, 0.46, '• Built-in CV', fontsize=7, ha='center', transform=ax.transAxes)
    plot_and_save(fig, 'A3_xgboost_architecture.png')

# ============================================================================
# 3. MODEL TRAINING FUNCTIONS
# ============================================================================

def hyperparameter_tuning(X_train, y_train, time_weights):
    log("\n" + "="*80)
    log("STEP 3: HYPERPARAMETER TUNING & MODEL SELECTION")
    log("="*80)
    
    time_weights_norm = time_weights / time_weights.max()
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train]) * time_weights_norm
    sample_weights = sample_weights / sample_weights.mean()
    
    tscv = TimeSeriesSplit(n_splits=5)
    tuning_results = {}
    best_models = {}
    
    # Random Forest
    log("\n  🔍 Tuning Random Forest...")
    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [8, 10], 'min_samples_split': [5, 8], 'min_samples_leaf': [2, 4]}
    rf_base = RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train, sample_weight=sample_weights)
    tuning_results['RandomForest'] = {'best_params': rf_grid.best_params_, 'best_score': rf_grid.best_score_, 'cv_results': rf_grid.cv_results_}
    best_models['RandomForest'] = rf_grid.best_estimator_
    log(f"    Best params: {rf_grid.best_params_}")
    log(f"    CV Score: {rf_grid.best_score_:.4f}")
    
    # Gradient Boosting
    log("\n  🔍 Tuning Gradient Boosting...")
    gb_param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.08], 'subsample': [0.8]}
    gb_base = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb_base, gb_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train, y_train, sample_weight=sample_weights)
    tuning_results['GradientBoosting'] = {'best_params': gb_grid.best_params_, 'best_score': gb_grid.best_score_, 'cv_results': gb_grid.cv_results_}
    best_models['GradientBoosting'] = gb_grid.best_estimator_
    log(f"    Best params: {gb_grid.best_params_}")
    log(f"    CV Score: {gb_grid.best_score_:.4f}")
    
    # XGBoost
    if XGB_AVAILABLE:
        log("\n  🔍 Tuning XGBoost...")
        xgb_param_grid = {'n_estimators': [100, 200], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.08], 'subsample': [0.8]}
        xgb_base = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        xgb_grid.fit(X_train, y_train, sample_weight=sample_weights)
        tuning_results['XGBoost'] = {'best_params': xgb_grid.best_params_, 'best_score': xgb_grid.best_score_, 'cv_results': xgb_grid.cv_results_}
        best_models['XGBoost'] = xgb_grid.best_estimator_
        log(f"    Best params: {xgb_grid.best_params_}")
        log(f"    CV Score: {xgb_grid.best_score_:.4f}")
    
    return tuning_results, best_models

def evaluate_models_comprehensive(models, X_test, y_test):
    log("\n" + "="*80)
    log("STEP 4: MODEL EVALUATION & COMPARISON")
    log("="*80)
    
    results = {}
    for model_name, model in models.items():
        log(f"\n  📊 Evaluating {model_name}...")
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrated.fit(X_test, y_test)
        y_pred = calibrated.predict(X_test)
        y_proba = calibrated.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        brier_scores = []
        for class_idx in [0, 1, 2]:
            y_true_binary = (y_test == class_idx).astype(int)
            brier = brier_score_loss(y_true_binary, y_proba[:, class_idx]) if len(np.unique(y_true_binary)) == 2 else 0.25
            brier_scores.append(brier)
        brier_avg = np.mean(brier_scores)
        
        roc_aucs = []
        for class_idx in [0, 1, 2]:
            y_true_binary = (y_test == class_idx).astype(int)
            roc_auc = roc_auc_score(y_true_binary, y_proba[:, class_idx]) if len(np.unique(y_true_binary)) == 2 else 0.5
            roc_aucs.append(roc_auc)
        roc_auc_avg = np.mean(roc_aucs)
        
        log_loss_val = log_loss(y_test, y_proba)
        
        results[model_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
                               'brier_score': brier_avg, 'roc_auc': roc_auc_avg, 'log_loss': log_loss_val,
                               'model': calibrated, 'base_model': model}
        log(f"    Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc_avg:.4f}, Brier: {brier_avg:.4f}")
    
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    log(f"\n🏆 BEST MODEL SELECTED: {best_model_name} (ROC-AUC={results[best_model_name]['roc_auc']:.4f})")
    return results, best_model_name

def train_best_model(X_train, y_train, time_weights, best_params, model_type):
    time_weights_norm = time_weights / time_weights.max()
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train]) * time_weights_norm
    sample_weights = sample_weights / sample_weights.mean()
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier(**best_params, random_state=42, class_weight=class_weight_dict, n_jobs=-1)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_type == 'XGBoost' and XGB_AVAILABLE:
        model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight=class_weight_dict, n_jobs=-1)
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated.fit(X_train, y_train, sample_weight=sample_weights)
    return calibrated, model

# ============================================================================
# 4. PREDICTION AND BETTING FUNCTIONS
# ============================================================================

def predict_match(model, team_encoder, home_team, away_team, month, season_stage,
                  odds_h, odds_d, odds_a, default_team='Arsenal'):
    try:
        home_enc = team_encoder.transform([home_team])[0]
        away_enc = team_encoder.transform([away_team])[0]
    except ValueError:
        home_enc = team_encoder.transform([default_team])[0]
        away_enc = team_encoder.transform([default_team])[0]
    
    home_implied = 1 / odds_h
    draw_implied = 1 / odds_d
    away_implied = 1 / odds_a
    total_implied = home_implied + draw_implied + away_implied
    
    home_norm = home_implied / total_implied
    draw_norm = draw_implied / total_implied
    away_norm = away_implied / total_implied
    
    odds_spread = max(odds_h, odds_d, odds_a) - min(odds_h, odds_d, odds_a)
    home_adv = home_norm - away_norm
    overround = total_implied - 1
    
    features = np.array([[
        month, home_enc, away_enc,
        0.35, 0.25, 0.5, 0.35, 0.25, 0.5, 0.35,
        odds_h, odds_d, odds_a,
        home_norm, draw_norm, away_norm,
        home_norm*odds_h, draw_norm*odds_d, away_norm*odds_a,
        odds_spread, season_stage, home_adv, overround
    ]])
    
    proba = model.predict_proba(features)[0]
    proba = proba / proba.sum()
    
    return {
        'home_prob': proba[2], 'draw_prob': proba[1], 'away_prob': proba[0],
        'home_market_prob': home_norm, 'draw_market_prob': draw_norm, 'away_market_prob': away_norm,
        'home_ev': proba[2] * (odds_h - 1) - (1 - proba[2]),
        'draw_ev': proba[1] * (odds_d - 1) - (1 - proba[1]),
        'away_ev': proba[0] * (odds_a - 1) - (1 - proba[0])
    }

def select_bets(predictions, total_bankroll=100, max_bets=3):
    def kelly(prob, odds, max_f=0.10):
        if odds <= 1 or prob <= 0:
            return 0
        k = (prob * odds - 1) / (odds - 1)
        return max(0, min(max_f, k))
    
    candidates = []
    for p in predictions:
        for outcome, ev_key, prob_key, odds_key in [('H', 'home_ev', 'home_prob', 'odds_h'),
                                                      ('D', 'draw_ev', 'draw_prob', 'odds_d'),
                                                      ('A', 'away_ev', 'away_prob', 'odds_a')]:
            ev = p[ev_key]
            if ev > 0.01:
                odds = p[odds_key]
                prob = p[prob_key]
                k = kelly(prob, odds)
                candidates.append({'match': p['match'], 'outcome': outcome, 'odds': odds,
                                   'ev': ev, 'prob': prob, 'kelly': k, 'score': ev * (1 + k)})
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected, used = [], set()
    for c in candidates:
        if c['match'] not in used and len(selected) < max_bets:
            selected.append(c)
            used.add(c['match'])
    
    total_k = sum(b['kelly'] for b in selected)
    if total_k > 0:
        for b in selected:
            b['allocation'] = (b['kelly'] / total_k) * total_bankroll
    else:
        for b in selected:
            b['allocation'] = total_bankroll / len(selected) if selected else 0
    return selected

def evaluate_betting_performance(selected_bets, actual_results):
    total_profit = 0
    total_stake = 0
    for bet in selected_bets:
        total_stake += bet['allocation']
        if actual_results.get(bet['match']) == bet['outcome']:
            total_profit += bet['allocation'] * (bet['odds'] - 1)
        else:
            total_profit -= bet['allocation']
    return {'total_stake': total_stake, 'total_profit': total_profit, 
            'roi': (total_profit/total_stake)*100 if total_stake>0 else 0, 
            'is_profitable': total_profit > 0}

def evaluate_model_by_betting(model, X_test_round, y_test_round, df_test_round, team_encoder):
    actual_results = {}
    for i in range(len(df_test_round)):
        match_key = f"{df_test_round.iloc[i]['HOMETEAM']} vs {df_test_round.iloc[i]['AWAYTEAM']}"
        result_map_rev = {2: 'H', 1: 'D', 0: 'A'}
        actual_results[match_key] = result_map_rev[y_test_round[i]]
    
    predictions = []
    for i in range(len(df_test_round)):
        row = df_test_round.iloc[i]
        match_key = f"{row['HOMETEAM']} vs {row['AWAYTEAM']}"
        pred = predict_match(model, team_encoder, row['HOMETEAM'], row['AWAYTEAM'], 
                             row['MONTH'], row['SEASON_STAGE'],
                             row['BET365H'], row['BET365D'], row['BET365A'])
        pred['match'] = match_key
        pred['odds_h'] = row['BET365H']; pred['odds_d'] = row['BET365D']; pred['odds_a'] = row['BET365A']
        predictions.append(pred)
    
    selected_bets = select_bets(predictions)
    performance = evaluate_betting_performance(selected_bets, actual_results)
    return {'selected_bets': selected_bets, 'actual_results': actual_results, 
            'total_stake': performance['total_stake'], 'total_profit': performance['total_profit'], 
            'roi': performance['roi'], 'is_profitable': performance['is_profitable']}

# ============================================================================
# 5. ROLLING MODIFICATION SIMULATION
# ============================================================================

def rolling_modification_simulation(df, team_encoder, all_teams, feature_cols, best_model_info,
                                     initial_train_size=1140, round_size=10):
    log("\n" + "="*80)
    log("STEP 5: ROLLING MODIFICATION SIMULATION")
    log("="*80)
    log(f"Initial training: {initial_train_size} matches")
    log(f"Round size: {round_size} matches")
    log(f"Best model: {best_model_info['name']}")
    
    historical_df = df[df['RESULT_NUM'].notna()].copy()
    X = historical_df[feature_cols].values
    y = historical_df['RESULT_NUM'].values
    w = historical_df['TIME_WEIGHT'].values
    total_matches = len(historical_df)
    log(f"Total historical matches: {total_matches}")
    
    round_metrics = []
    feature_importance_history = []
    all_betting_records = []
    profit_history = []
    
    train_size = initial_train_size
    current_model, current_rf = train_best_model(X[:train_size], y[:train_size], w[:train_size],
                                                   best_model_info['params'], best_model_info['type'])
    
    if hasattr(current_rf, 'feature_importances_'):
        imp = pd.DataFrame({'feature': feature_cols, 'importance': current_rf.feature_importances_}).sort_values('importance', ascending=False).head(10)
        feature_importance_history.append({'round': 0, 'importance': imp, 'train_size': train_size})
    
    pos = train_size
    round_num = 1
    cum_profit = 0
    
    log(f"\n{'='*110}")
    log(f"{'Round':<6} {'Test Range':<20} {'Net Profit':<12} {'ROI':<10} {'Profitable':<10} {'Cumulative':<12}")
    log(f"{'='*110}")
    
    while pos + round_size <= total_matches:
        X_test = X[pos:pos+round_size]
        y_test = y[pos:pos+round_size]
        df_test = historical_df.iloc[pos:pos+round_size]
        
        try:
            metrics = evaluate_model_by_betting(current_model, X_test, y_test, df_test, team_encoder)
        except Exception as e:
            log(f"  Round {round_num} failed: {e}")
            pos += round_size
            round_num += 1
            continue
        
        metrics['round'] = round_num
        metrics['test_range'] = f"{pos}-{pos+round_size-1}"
        round_metrics.append(metrics)
        
        cum_profit += metrics['total_profit']
        profit_history.append({'round': round_num, 'round_profit': metrics['total_profit'], 
                               'cumulative_profit': cum_profit, 'roi': metrics['roi']})
        
        for bet in metrics['selected_bets']:
            actual = metrics['actual_results'].get(bet['match'], '?')
            all_betting_records.append({'Round': round_num, 'Match': bet['match'], 'Bet': bet['outcome'],
                                        'Odds': bet['odds'], 'Stake': bet['allocation'], 'Actual': actual,
                                        'Result': 'WIN' if actual == bet['outcome'] else 'LOSS',
                                        'Round_Profit': bet['allocation']*(bet['odds']-1) if actual == bet['outcome'] else -bet['allocation']})
        
        log(f"{round_num:<6} {metrics['test_range']:<20} ¥{metrics['total_profit']:<11.2f} {metrics['roi']:<9.2f}% {'✅' if metrics['is_profitable'] else '❌':<10} ¥{cum_profit:<11.2f}")
        
        # Retrain with expanded data
        train_size += round_size
        if train_size <= total_matches:
            current_model, current_rf = train_best_model(X[:train_size], y[:train_size], w[:train_size],
                                                           best_model_info['params'], best_model_info['type'])
            
            if hasattr(current_rf, 'feature_importances_'):
                imp = pd.DataFrame({'feature': feature_cols, 'importance': current_rf.feature_importances_}).sort_values('importance', ascending=False).head(10)
                feature_importance_history.append({'round': round_num, 'importance': imp, 'train_size': train_size})
        
        pos += round_size
        round_num += 1
    
    log(f"\n{'='*110}")
    log(f"SIMULATION COMPLETED")
    log(f"  Total rounds processed: {len(round_metrics)}")
    log(f"  Final training size: {train_size} matches")
    log(f"  Cumulative Net Profit: ¥{cum_profit:.2f}")
    log("="*110)
    
    return round_metrics, feature_importance_history, profit_history, all_betting_records

# ============================================================================
# 6. CAPITAL METRICS (NO IRR)
# ============================================================================

def calculate_capital_metrics(profit_history, round_metrics):
    """Calculate capital efficiency metrics including profitable rounds"""
    if not profit_history:
        return {'max_capital': 0, 'payback_round': None, 'payback_days': None,
                'total_rounds': 0, 'cumulative_profit': 0, 'profitable_rounds': 0,
                'profit_rate': 0}
    
    cumulative = [p['cumulative_profit'] for p in profit_history]
    max_capital = max(0, -min(cumulative))
    
    payback_round = None
    for i, c in enumerate(cumulative):
        if c >= 0:
            payback_round = i + 1
            break
    
    total_rounds = len(round_metrics)
    cumulative_profit = cumulative[-1] if cumulative else 0
    profitable_rounds = sum(1 for m in round_metrics if m['is_profitable'])
    profit_rate = (profitable_rounds / total_rounds * 100) if total_rounds > 0 else 0
    
    return {'max_capital': max_capital, 'payback_round': payback_round, 
            'payback_days': payback_round * 10 if payback_round else None,
            'total_rounds': total_rounds, 'cumulative_profit': cumulative_profit,
            'profitable_rounds': profitable_rounds, 'profit_rate': profit_rate}

# ============================================================================
# 7. VISUALIZATIONS (NO ROI vs ACCURACY)
# ============================================================================

def plot_and_save(fig, filename):
    """Helper function to display and save a figure"""
    filepath = os.path.join(SAVE_PATH, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.5)
    log(f"✓ Saved and displayed: {filepath}")

def plot_model_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.25
    ax1 = axes[0]
    metrics = ['accuracy', 'roc_auc', 'f1_score']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [results[n][m] for n in names]
        ax1.bar(x + (i-1)*width, vals, width, label=m.upper(), color=c, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x); ax1.set_xticklabels(names); ax1.set_ylim(0,1); ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison', fontweight='bold'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    brier = [results[n]['brier_score'] for n in names]
    bars = ax2.bar(names, brier, color=['#3498db','#2ecc71','#e74c3c'][:len(names)], edgecolor='black')
    ax2.set_ylabel('Brier Score (lower is better)'); ax2.set_title('Model Calibration Quality', fontweight='bold')
    for bar, val in zip(bars, brier):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.4f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3, axis='y')
    plot_and_save(fig, '1_model_comparison.png')

def plot_hyperparameter_tuning(tuning_results):
    fig, axes = plt.subplots(1, len(tuning_results), figsize=(16, 5))
    if len(tuning_results) == 1: axes = [axes]
    for idx, (model_name, results) in enumerate(tuning_results.items()):
        ax = axes[idx]
        cv_results = results['cv_results']
        param_name = list(results['best_params'].keys())[0] if results['best_params'] else 'n_estimators'
        param_vals = cv_results[f'param_{param_name}'].data
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        try: param_numeric = [float(p) for p in param_vals]
        except: param_numeric = range(len(param_vals))
        ax.errorbar(param_numeric, mean_scores, yerr=std_scores, fmt='o-', capsize=5, color='#3498db', markersize=8)
        ax.set_xlabel(param_name); ax.set_ylabel('CV Accuracy'); ax.set_title(f'{model_name} - Tuning', fontweight='bold')
        ax.grid(True, alpha=0.3)
        best_param = results['best_params'].get(param_name)
        if best_param:
            try: best_idx = list(param_vals).index(best_param); ax.scatter(param_numeric[best_idx], mean_scores[best_idx], color='red', s=150, marker='*')
            except: pass
    plot_and_save(fig, '2_hyperparameter_tuning.png')

def plot_profit_over_rounds(profit_history):
    if not profit_history: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rounds = [p['round'] for p in profit_history]
    profits = [p['round_profit'] for p in profit_history]
    cumulative = [p['cumulative_profit'] for p in profit_history]
    ax1 = axes[0]
    colors = ['#2ecc71' if p>0 else '#e74c3c' for p in profits]
    ax1.bar(rounds, profits, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black'); ax1.set_xlabel('Round'); ax1.set_ylabel('Net Profit (¥)')
    ax1.set_title('Profit/Loss per Round', fontweight='bold'); ax1.grid(True, alpha=0.3, axis='y')
    ax2 = axes[1]
    ax2.plot(rounds, cumulative, 'b-o', linewidth=2, markersize=8)
    ax2.fill_between(rounds, 0, cumulative, where=(np.array(cumulative)>0), color='#2ecc71', alpha=0.3)
    ax2.fill_between(rounds, 0, cumulative, where=(np.array(cumulative)<0), color='#e74c3c', alpha=0.3)
    ax2.axhline(y=0, color='black'); ax2.set_xlabel('Round'); ax2.set_ylabel('Cumulative Profit (¥)')
    ax2.set_title('Cumulative Profit Over Rounds', fontweight='bold'); ax2.grid(True, alpha=0.3)
    plot_and_save(fig, '3_profit_over_rounds.png')

def plot_feature_evolution(feature_importance_history, feature_cols, top_n=10):
    if len(feature_importance_history) < 2: return
    rounds = [h['round'] for h in feature_importance_history]
    final_imp = feature_importance_history[-1]['importance']
    top = final_imp['feature'].head(top_n).tolist()
    matrix = []
    for feat in top:
        vals = []
        for h in feature_importance_history:
            d = dict(zip(h['importance']['feature'], h['importance']['importance']))
            vals.append(d.get(feat, 0))
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(14,8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.15)
    ax.set_xticks(range(len(rounds))); ax.set_xticklabels([f"R{r}" for r in rounds], rotation=45, ha='right')
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=10)
    ax.set_xlabel('Rolling Round'); ax.set_ylabel('Top Features')
    ax.set_title('Top 10 Feature Importance Evolution', fontweight='bold')
    plt.colorbar(im, label='Importance')
    plot_and_save(fig, '4_feature_evolution.png')

def plot_final_features(feature_importance_history):
    if not feature_importance_history: return
    final = feature_importance_history[-1]['importance']
    features = final['feature'].head(10).values[::-1]
    values = final['importance'].head(10).values[::-1]
    fig, ax = plt.subplots(figsize=(12,8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(values)))
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(values))); ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Importance'); ax.set_title('Top 10 Feature Importance (Final)', fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis='x')
    plot_and_save(fig, '5_final_features.png')

def plot_capital_metrics(capital_metrics):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Format numbers with commas
    max_cap = f"¥{capital_metrics['max_capital']:,.2f}"
    cum_profit = f"¥{capital_metrics['cumulative_profit']:,.2f}"
    payback = f"{capital_metrics['payback_round']} rounds ({capital_metrics['payback_days']} days)" if capital_metrics['payback_round'] else "Not achieved"
    profit_rate = f"{capital_metrics['profit_rate']:.1f}%"
    
    text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   CAPITAL EFFICIENCY METRICS                 ║
    ╚══════════════════════════════════════════════════════════════╝
    
        Total Rounds:              {capital_metrics['total_rounds']}
        
        Cumulative Profit:         {cum_profit}
        
        Maximum Capital Required:  {max_cap}
        
        Payback Period:            {payback}
        
        Profitable Rounds:         {capital_metrics['profitable_rounds']} / {capital_metrics['total_rounds']}
        
        Profit Rate:               {profit_rate}
    
    ╔══════════════════════════════════════════════════════════════╗
    ║  Maximum Capital Required = Peak loss during simulation    ║
    ║  Payback Period = Time to recover initial capital          ║
    ║  Profit Rate = (Profitable Rounds / Total Rounds) × 100%   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=12, verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plot_and_save(fig, '6_capital_metrics.png')

def plot_betting_summary(selected_bets):
    if not selected_bets: return
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    ax1 = axes[0]
    labels = [f"{b['match'][:15]}\n{b['outcome']} @ {b['odds']:.2f}" for b in selected_bets]
    sizes = [b['allocation'] for b in selected_bets]
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(selected_bets)]
    ax1.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, explode=[0.05]*len(selected_bets), shadow=True)
    ax1.set_title('Next Round - Stake Allocation (¥100)', fontweight='bold')
    ax2 = axes[1]
    names = [f"{b['match'][:12]}\n{b['outcome']}" for b in selected_bets]
    evs = [b['ev']*100 for b in selected_bets]
    colors_ev = ['#2ecc71' if ev>0 else '#e74c3c' for ev in evs]
    bars = ax2.bar(names, evs, color=colors_ev, edgecolor='black')
    ax2.axhline(y=0, color='black'); ax2.set_ylabel('Expected Return (%)')
    ax2.set_title('Next Round - Expected Value by Bet', fontweight='bold')
    for bar, ev in zip(bars, evs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(0.5 if ev>0 else -2), f'{ev:+.1f}%', ha='center', va='bottom' if ev>0 else 'top', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plot_and_save(fig, '7_next_round_summary.png')

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    log("="*80)
    log("PREMIER LEAGUE BETTING STRATEGY - COMPLETE MODEL")
    log(f"Files will be saved to: {SAVE_PATH}")
    log("="*80)
    
    try:
        # Generate architecture diagrams
        log("\n" + "="*80)
        log("GENERATING MODEL ARCHITECTURE DIAGRAMS")
        log("="*80)
        plot_random_forest_architecture()
        plot_gradient_boosting_architecture()
        plot_xgboost_architecture()
        
        # Load data from GitHub
        df = load_and_preprocess_data()
        team_encoder, all_teams = create_team_encoders(df)
        df_featured = create_features(df, team_encoder, all_teams)
        
        feature_cols = ['MONTH', 'HOME_TEAM_ENCODED', 'AWAY_TEAM_ENCODED',
                        'HOME_LAST5_WIN_RATE', 'HOME_LAST5_DRAW_RATE', 'HOME_MOMENTUM',
                        'AWAY_LAST5_WIN_RATE', 'AWAY_LAST5_DRAW_RATE', 'AWAY_MOMENTUM',
                        'HEAD_TO_HEAD_HOME_WIN_RATE', 'BET365H', 'BET365D', 'BET365A',
                        'HOME_NORM_PROB', 'DRAW_NORM_PROB', 'AWAY_NORM_PROB',
                        'HOME_VALUE', 'DRAW_VALUE', 'AWAY_VALUE', 'ODDS_SPREAD',
                        'SEASON_STAGE', 'HOME_ADVANTAGE', 'BOOKMAKER_OVERROUND']
        
        # Model selection on historical data
        historical_df = df_featured[df_featured['RESULT_NUM'].notna()].copy()
        X = historical_df[feature_cols].values
        y = historical_df['RESULT_NUM'].values
        w = historical_df['TIME_WEIGHT'].values
        
        X_init = X[:1140]; y_init = y[:1140]; w_init = w[:1140]
        X_val = X[1140:1240]; y_val = y[1140:1240]
        
        tuning_results, best_models = hyperparameter_tuning(X_init, y_init, w_init)
        eval_results, best_name = evaluate_models_comprehensive(best_models, X_val, y_val)
        
        best_model_info = {'name': best_name, 'type': best_name, 'params': tuning_results[best_name]['best_params']}
        
        log(f"\nBEST MODEL: {best_name}")
        log(f"  Parameters: {best_model_info['params']}")
        
        # Rolling simulation
        round_metrics, feature_importance_history, profit_history, betting_records = rolling_modification_simulation(
            df_featured, team_encoder, all_teams, feature_cols, best_model_info)
        
        # Capital metrics (includes profitable rounds info)
        capital_metrics = calculate_capital_metrics(profit_history, round_metrics)
        
        # Predict next round
        upcoming = df_featured[df_featured['RESULT_NUM'].isna()].copy()
        if len(upcoming) > 0:
            X_all = historical_df[feature_cols].values
            y_all = historical_df['RESULT_NUM'].values
            w_all = historical_df['TIME_WEIGHT'].values
            final_model, _ = train_best_model(X_all, y_all, w_all, best_model_info['params'], best_model_info['type'])
            
            log("\n" + "="*80)
            log("PREDICTING NEXT ROUND")
            log("="*80)
            
            predictions = []
            for _, row in upcoming.iterrows():
                pred = predict_match(final_model, team_encoder, row['HOMETEAM'], row['AWAYTEAM'],
                                     row['MONTH'], row['SEASON_STAGE'],
                                     row['BET365H'], row['BET365D'], row['BET365A'])
                pred['match'] = f"{row['HOMETEAM']} vs {row['AWAYTEAM']}"
                pred['odds_h'] = row['BET365H']; pred['odds_d'] = row['BET365D']; pred['odds_a'] = row['BET365A']
                predictions.append(pred)
                log(f"\n{row['HOMETEAM']} vs {row['AWAYTEAM']}")
                log(f"  Odds: H={row['BET365H']:.2f} D={row['BET365D']:.2f} A={row['BET365A']:.2f}")
                log(f"  Model: H={pred['home_prob']:.1%} D={pred['draw_prob']:.1%} A={pred['away_prob']:.1%}")
                log(f"  EV: H={pred['home_ev']:+.3f} D={pred['draw_ev']:+.3f} A={pred['away_ev']:+.3f}")
            
            selected_bets = select_bets(predictions)
            log("\n" + "="*80)
            log("RECOMMENDED BETS FOR NEXT ROUND")
            log("="*80)
            for b in selected_bets:
                log(f"  {b['match']}: {b['outcome']} @ {b['odds']:.2f} (Confidence: {b['prob']:.1%}, Stake: ¥{b['allocation']:.1f})")
            
            plot_betting_summary(selected_bets)
        
        # Generate output charts
        log("\n" + "="*80)
        log("GENERATING OUTPUT CHARTS")
        log("="*80)
        plot_model_comparison(eval_results)
        plot_hyperparameter_tuning(tuning_results)
        plot_profit_over_rounds(profit_history)
        if feature_importance_history:
            plot_feature_evolution(feature_importance_history, feature_cols)
            plot_final_features(feature_importance_history)
        plot_capital_metrics(capital_metrics)
        
        # Summary
        log("\n" + "="*80)
        log("FINAL SUMMARY")
        log("="*80)
        log(f"  Total Rounds: {capital_metrics['total_rounds']}")
        log(f"  Cumulative Profit: ¥{capital_metrics['cumulative_profit']:.2f}")
        log(f"  Max Capital Required: ¥{capital_metrics['max_capital']:.2f}")
        log(f"  Payback Period: {capital_metrics['payback_round']} rounds ({capital_metrics['payback_days']} days)" if capital_metrics['payback_round'] else "  Payback Period: Not achieved")
        log(f"  Profitable Rounds: {capital_metrics['profitable_rounds']}/{capital_metrics['total_rounds']} ({capital_metrics['profit_rate']:.1f}%)")
        
        log("\n" + "="*80)
        log("OUTPUT FILES")
        log("="*80)
        log(f"  {os.path.join(SAVE_PATH, 'A1_random_forest_architecture.png')}")
        log(f"  {os.path.join(SAVE_PATH, 'A2_gradient_boosting_architecture.png')}")
        log(f"  {os.path.join(SAVE_PATH, 'A3_xgboost_architecture.png')}")
        log(f"  {os.path.join(SAVE_PATH, '1_model_comparison.png')}")
        log(f"  {os.path.join(SAVE_PATH, '2_hyperparameter_tuning.png')}")
        log(f"  {os.path.join(SAVE_PATH, '3_profit_over_rounds.png')}")
        log(f"  {os.path.join(SAVE_PATH, '4_feature_evolution.png')}")
        log(f"  {os.path.join(SAVE_PATH, '5_final_features.png')}")
        log(f"  {os.path.join(SAVE_PATH, '6_capital_metrics.png')}")
        log(f"  {os.path.join(SAVE_PATH, '7_next_round_summary.png')}")
        
        log("\n" + "="*80)
        log("DISCLAIMER")
        log("="*80)
        log("This analysis is for educational purposes only. Sports betting involves")
        log("significant risk of loss. Past performance does not guarantee future results.")
        
        # Keep the last figure window open
        plt.show(block=True)
        
    except Exception as e:
        log(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()