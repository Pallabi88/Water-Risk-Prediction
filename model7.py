import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def norm(col, value):
    x = (
        (value - df_train[col].min()) /
        (df_train[col].max() - df_train[col].min() + 1e-6)
    )
    return float(np.clip(x, 0, 1))

def _risk_prob(model, scaler, X_cols, row_dict):
    import pandas as pd
    df = pd.DataFrame([row_dict])
    df_scaled = scaler.transform(df[X_cols])
    return float(model.predict_proba(df_scaled)[0][1])

# =============================================================================
# STEP 0 - LOAD DATASET
# =============================================================================

def load_data(file_path='WaterSystem_Dataset.xlsx'):
    df = pd.read_excel(file_path)
    print('='*80)
    print('SMART WATER SUPPLY DISRUPTION AND SHORTAGE PREDICTION')
    print('='*80)
    print(f'Dataset size: {df.shape}')
    print(df.head())
    print()
    return df


# =============================================================================
# STEP 1 - DATA CLEANING
# =============================================================================

def clean_data(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    int_cols = ['Population', 'Pipe_Age', 'Maintenance_History', 'Leakage_History']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    print('STEP 1 - Data Cleaning Done')
    print(f'Duplicates removed: {removed}')
    print()

    return df

#===============LOGIC===========================#
def rebuild_target(df):
    df = df.copy()

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    # =========================
    # BASE STRESSES
    # =========================
    rainfall_stress = 1 - norm(df['Rainfall(mm)'])
    demand_stress = norm(df['Population_Demand'])
    flow_stress = 1 - norm(df['Flow_Rate(L/min)'])
    temp_stress = norm(df['Temperature(C)'])

    leakage_stress = norm(df['Leakage_History'])
    pressure_stress = 1 - norm(df['Water_Pressure(psi)'])
    age_stress = norm(df['Pipe_Age'])
    maintenance_stress = 1 - norm(df['Maintenance_History'])

    ph_dev = abs(df['pH_Level'] - 7)
    quality_stress = norm(ph_dev)

    # =========================
    # NONLINEAR INTERACTIONS
    # =========================

    # climate
    drought_heat = rainfall_stress * temp_stress
    severe_climate = rainfall_stress * temp_stress * flow_stress

    # demand overload
    demand_flow = demand_stress * flow_stress
    demand_explosion = demand_stress ** 2

    # infrastructure decay
    leak_age = leakage_stress * age_stress
    leak_age_damage = (leak_age) ** 2

    # maintenance amplification
    pressure_maintenance = pressure_stress * maintenance_stress
    infra_decay = (
        leakage_stress *
        age_stress *
        maintenance_stress
    )

    # contamination concentration
    quality_flow = quality_stress * flow_stress

    # threshold jump effects
    critical_pressure = (pressure_stress > 0.72).astype(float)
    critical_flow = (flow_stress > 0.75).astype(float)
    extreme_heat = (temp_stress > 0.80).astype(float)

    # cascading urban collapse
    cascade_failure = (
        leakage_stress *
        pressure_stress *
        age_stress *
        maintenance_stress
    )

    # multi-system crisis
    full_system_stress = (
        rainfall_stress *
        demand_stress *
        leakage_stress *
        pressure_stress
    )

    # =========================
    # FINAL ADVANCED RISK SCORE
    # =========================
    total_score = (
        # base
        0.10 * rainfall_stress +
        0.07 * demand_stress +
        0.06 * flow_stress +
        0.05 * temp_stress +
        0.08 * leakage_stress +
        0.06 * pressure_stress +
        0.05 * age_stress +
        0.05 * maintenance_stress +
        0.03 * quality_stress +

        # nonlinear
        0.08 * drought_heat +
        0.05 * severe_climate +
        0.05 * demand_flow +
        0.04 * demand_explosion +
        0.05 * leak_age +
        0.04 * leak_age_damage +
        0.04 * pressure_maintenance +
        0.04 * infra_decay +
        0.03 * quality_flow +
        0.03 * cascade_failure +
        0.03 * full_system_stress +

        # threshold jumps
        0.03 * critical_pressure +
        0.02 * critical_flow +
        0.01 * extreme_heat
    )

    #noise = np.random.normal(
    #    loc=0,
    #    scale=0.01,
    #    size=len(df)
    #)

    #total_score = total_score + noise

    threshold = total_score.quantile(0.62)

    df['Water_Risk'] = (
        total_score > threshold
    ).astype(int)

    print('STEP 2 - Advanced Nonlinear Water_Risk Rebuilt')
    print(df['Water_Risk'].value_counts())
    print()

    return df




# =============================================================================
# STEP 2 - FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    df = df.copy()

    df['Demand_Supply_Ratio'] = (
        df['Population_Demand'] /
        (df['Flow_Rate(L/min)'] + 1)
    )

    df['Pressure_Efficiency'] = (
        df['Water_Pressure(psi)'] *
        df['Pipe_Diameter(mm)']
    ) / (df['Pipe_Age'] + 1)

    df['Climate_Stress'] = (
        df['Temperature(C)'] /
        (df['Rainfall(mm)'] + 1)
    )

    df['pH_Deviation'] = abs(df['pH_Level'] - 7)

    print('STEP 3 - Feature Engineering Done')
    print(f'Total features: {df.shape[1]-1}')
    print()

    return df

# =============================================================================
# FEATURE DISTRIBUTION VISUALIZATION
# =============================================================================
def plot_feature_histograms(df):
    import matplotlib.pyplot as plt

    cols = [
        'Rainfall(mm)',
        'Temperature(C)',
        'Soil_Type',
        'Population',
        'Population_Demand',
        'Pipe_Age',
        'Pipe_Diameter(mm)',
        'Water_Pressure(psi)',
        'Flow_Rate(L/min)',
        'pH_Level',
        'Leakage_History',
        'Maintenance_History',
        'Water_Risk',
        'Demand_Supply_Ratio',
        'Pressure_Efficiency',
        'Climate_Stress',
        'pH_Deviation'
    ]

    plt.figure(figsize=(15, 14))

    for i, col in enumerate(cols, 1):
        plt.subplot(5, 4, i)

        plt.hist(
            df[col],
            bins=12,
            edgecolor='black'
        )

        plt.title(
            col,
            fontsize=9,
            fontweight='bold'
        )

        plt.grid(True, alpha=0.35)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout(pad=2.0)
    plt.show()


# =============================================================================
# STEP 3 - SPLIT & SCALE
# =============================================================================

def prepare_data(df):
    # remove identifier columns
    X = df.drop(
        ['Water_Risk'],
        axis=1
    )

    y = df['Water_Risk']

    print('Class Distribution:')
    print(y.value_counts())
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print('STEP 3 - Data Splitting & Scaling Done')
    print(f'Train size: {len(X_train)}')
    print(f'Test size : {len(X_test)}')
    print()

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# =============================================================================
# STEP 4 - MODEL TRAINING (INCLUDING XGBOOST)
# =============================================================================

def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=4,
    random_state=42
),

'Random Forest': RandomForestClassifier(
    n_estimators=180,
    min_samples_leaf=3,
    random_state=42
),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=130,
    random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    eval_metric='logloss',
    random_state=42
        )
    }

    trained_models = {}

    print('STEP 4 - Model Training')

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        print(f'{name:<22} Train: {train_acc*100:.2f}%   Test: {test_acc*100:.2f}%')

    print()

    return trained_models

# =============================================================================
# STEP 5 - ADVANCED MODEL COMPARISON
# =============================================================================

def compare_models(trained_models, X_train_scaled, y_train, X_test_scaled, y_test):
    results = []

    for name, model in trained_models.items():
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1 Score': f1_score(y_test, y_test_pred),
            'Overfit Gap': train_acc - test_acc
        })

    results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)

    print('STEP 5 - ADVANCED MODEL COMPARISON')
    print(results_df)
    print()

    return results_df


# =============================================================================
# STEP 6 - HYPERPARAMETER TUNING
# =============================================================================

def tune_top_models(results_df, X_train_scaled, y_train):
    top_models = results_df.head(3)['Model'].values

    print('Top Models Selected for Tuning:', top_models)
    print()

    tuned_models = {}

    for model_name in top_models:

        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10]
            }
            base_model = RandomForestClassifier(random_state=42)

        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4]
            }
            base_model = GradientBoostingClassifier(random_state=42)

        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4],
                'learning_rate': [0.05, 0.1]
            }
            base_model = XGBClassifier(eval_metric='logloss', random_state=42)

        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [3, 5, 10]}
            base_model = DecisionTreeClassifier(random_state=42)

        else:
            param_grid = {'C': [0.1, 1, 10]}
            base_model = LogisticRegression(max_iter=1000)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )

        grid.fit(X_train_scaled, y_train)

        tuned_models[model_name] = grid.best_estimator_

        print(f'Tuned {model_name}')
        print('Best Params:', grid.best_params_)
        print()

    return tuned_models


# =============================================================================
# STEP 7 - CROSS VALIDATION
# =============================================================================

def cross_validate_models(tuned_models, X_train_scaled, y_train):
    cv_results = []

    for name, model in tuned_models.items():
        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5,
            scoring='f1'
        )

        cv_results.append({
            'Model': name,
            'CV F1 Mean': scores.mean(),
            'CV Std': scores.std()
        })

    cv_df = pd.DataFrame(cv_results).sort_values('CV F1 Mean', ascending=False)

    print('STEP 7 - CROSS VALIDATION RESULTS')
    print(cv_df)
    print()

    final_model_name = cv_df.iloc[0]['Model']
    best_model = tuned_models[final_model_name]

    print('FINAL SELECTED MODEL:', final_model_name)
    print()

    return best_model, final_model_name, cv_df


# =============================================================================
# STEP 8 - FINAL EVALUATION
# =============================================================================

def final_evaluation(best_model, X_test_scaled, y_test):
    y_pred = best_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Risk', 'Risk'])

    print('STEP 8 - FINAL EVALUATION')
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(report)

    return y_pred, cm, report

# =============================================================================
# CONFUSION MATRIX PLOT (TOP 3 MODELS)
# =============================================================================
# =============================================================================
# SEPARATE CONFUSION MATRIX FIGURES
# =============================================================================
def plot_confusion_matrix_single(model, model_name, X_test_scaled, y_test):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['No Risk', 'Risk'],
        yticklabels=['No Risk', 'Risk']
    )

    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 9 - ROC / AUC ANALYSIS
# =============================================================================

def plot_roc_curves(models_dict, X_test_scaled, y_test):
    plt.figure(figsize=(10, 7))

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], '--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 10 - PRECISION RECALL CURVE
# =============================================================================

def plot_precision_recall(models_dict, X_test_scaled, y_test):
    plt.figure(figsize=(10, 7))

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AUC={pr_auc:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 11 - FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(best_model, X):
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        fi = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(fi['Feature'], fi['Importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()


# =============================================================================
# STEP 12 - SHAP EXPLAINABILITY
# =============================================================================

def run_shap_analysis(best_model, X_train, X_test):
    try:
        import shap

        print('Running SHAP Analysis...')

        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test[:200])

        shap.plots.bar(shap_values, max_display=15)
        shap.plots.beeswarm(shap_values, max_display=15)

    except Exception as e:
        print('SHAP skipped:', e)


# =============================================================================
# STEP 13 - SCENARIO SIMULATION ENGINE
# =============================================================================

def recompute_engineered_features(df_row):
    df_row = df_row.copy()

    df_row['Demand_Supply_Ratio'] = (
        df_row['Population_Demand'] /
        (df_row['Flow_Rate(L/min)'] + 1)
    )

    df_row['Pressure_Efficiency'] = (
        df_row['Water_Pressure(psi)'] *
        df_row['Pipe_Diameter(mm)']
    ) / (df_row['Pipe_Age'] + 1)

    df_row['Climate_Stress'] = (
        df_row['Temperature(C)'] /
        (df_row['Rainfall(mm)'] + 1)
    )

    df_row['pH_Deviation'] = abs(
        df_row['pH_Level'] - 7
    )

    return df_row


def simulate_scenarios(best_model, scaler, X):
    print('STEP 13 - ADVANCED SCENARIO SIMULATION')

    base = X.mean().to_dict()

    scenario_defs = {
        'Normal Condition': {},
        'Low Rainfall': {'Rainfall(mm)': 0.4},
        'Severe Drought': {'Rainfall(mm)': 0.15},
        'High Demand': {'Population_Demand': 1.4},
        'Extreme Demand': {'Population_Demand': 1.8},
        'Extreme Leakage': {'Leakage_History': 5},
        'Pipe Burst': {'Leakage_History': 6, 'Water_Pressure(psi)': 0.5},
        'Low Pressure': {'Water_Pressure(psi)': 0.6},
        'Pressure Collapse': {'Water_Pressure(psi)': 0.3},
        'Aging Pipe': {'Pipe_Age': 1.6},
        'Very Old Pipe': {'Pipe_Age': 2.0},
        'Poor pH': {'pH_Level': 5.5},
        'Contamination': {'pH_Level': 4.8},
        'Heat Stress': {'Temperature(C)': 1.25},
        'Extreme Heat': {'Temperature(C)': 1.45},
        'Low Flow': {'Flow_Rate(L/min)': 0.6},
        'Very Low Flow': {'Flow_Rate(L/min)': 0.35},
        'Poor Maintenance': {'Maintenance_History': 1},
        'No Maintenance': {'Maintenance_History': 0},
        'Demand + Leakage': {'Population_Demand': 1.5, 'Leakage_History': 5},
        'Drought + Heat': {'Rainfall(mm)': 0.2, 'Temperature(C)': 1.35},
        'Old Pipe + Leakage': {'Pipe_Age': 1.7, 'Leakage_History': 5},
        'Pressure + Flow Failure': {'Water_Pressure(psi)': 0.4, 'Flow_Rate(L/min)': 0.4},
        'Multi-System Stress': {'Rainfall(mm)': 0.25, 'Population_Demand': 1.5, 'Leakage_History': 5},
        'Critical Urban Failure': {'Rainfall(mm)': 0.1, 'Water_Pressure(psi)': 0.25, 'Leakage_History': 6}
    }

    names, risks = [], []

    for name, changes in scenario_defs.items():
        scenario = base.copy()

        for feature, value in changes.items():
            if feature in scenario:
                if isinstance(value, (int, float)) and value <= 3 and feature not in ['Leakage_History', 'Maintenance_History', 'pH_Level']:
                    scenario[feature] *= value
                else:
                    scenario[feature] = value

        scenario_df = pd.DataFrame([scenario])
        scenario_df = recompute_engineered_features(scenario_df)
        scenario_scaled = scaler.transform(scenario_df[X.columns])

        prob = best_model.predict_proba(scenario_scaled)[0][1]

        names.append(name)
        risks.append(prob)
        print(f'{name:<24} Risk Probability: {prob:.4f}')

    plt.figure(figsize=(15, 7))
    plt.bar(names, risks)
    plt.xticks(rotation=65, ha='right')
    plt.ylabel('Risk Probability')
    plt.title('Scenario-wise Water Disruption Risk Prediction')
    plt.tight_layout()
    plt.show()


def diagnose_and_optimize(df, best_model, scaler, X):
    import numpy as np
    import pandas as pd

    print('\n' + '='*95)
    print('STEP 14 - ROOT CAUSE DIAGNOSIS & OPTIMIZATION ENGINE')
    print('='*95)

    # ==========================================================
    # BASE SYSTEM STATE
    # ==========================================================
    base = df.mean(numeric_only=True).to_dict()

    # engineered features
    base['Demand_Supply_Ratio'] = (
        base['Population_Demand'] /
        (base['Flow_Rate(L/min)'] + 1)
    )

    base['Pressure_Efficiency'] = (
        base['Water_Pressure(psi)'] *
        base['Pipe_Diameter(mm)']
    ) / (base['Pipe_Age'] + 1)

    base['Climate_Stress'] = (
        base['Temperature(C)'] /
        (base['Rainfall(mm)'] + 1)
    )

    base['pH_Deviation'] = abs(base['pH_Level'] - 7)

    X_cols = list(X.columns)

    # current prediction
    current_prob = _risk_prob(best_model, scaler, X_cols, base)
    risk_label = 'YES' if current_prob >= 0.5 else 'NO'

    print(f'\nPredicted Water Risk : {risk_label}')
    print(f'Risk Probability     : {current_prob:.4f}\n')

    if risk_label == 'NO':
        print('System Status: Healthy')
        print('Recommendation: Routine monitoring only.\n')
        return

    # ==========================================================
    # NORMALIZATION HELPER
    # ==========================================================
    def norm(col, value):
        return float(
            (value - df[col].min()) /
            (df[col].max() - df[col].min() + 1e-6)
        )

    # ==========================================================
    # ROOT CAUSE SCORES
    # ==========================================================

    rainfall_stress = 1 - norm('Rainfall(mm)', base['Rainfall(mm)'])
    demand_stress = norm('Population_Demand', base['Population_Demand'])
    flow_stress = 1 - norm('Flow_Rate(L/min)', base['Flow_Rate(L/min)'])
    temp_stress = norm('Temperature(C)', base['Temperature(C)'])

    leakage_stress = norm('Leakage_History', base['Leakage_History'])
    pressure_stress = 1 - norm('Water_Pressure(psi)', base['Water_Pressure(psi)'])
    age_stress = norm('Pipe_Age', base['Pipe_Age'])
    maintenance_stress = 1 - norm('Maintenance_History', base['Maintenance_History'])

    quality_stress = norm(
        'pH_Level',
        abs(base['pH_Level'] - 7)
    )

    # shortage score
    shortage_score = (
        0.35 * rainfall_stress +
        0.30 * demand_stress +
        0.20 * flow_stress +
        0.15 * temp_stress
    )

    # disruption score
    disruption_score = (
        0.35 * leakage_stress +
        0.25 * pressure_stress +
        0.20 * age_stress +
        0.20 * maintenance_stress
    )

    # quality score
    quality_score = quality_stress

    scores = {
        'SHORTAGE': shortage_score,
        'DISRUPTION': disruption_score,
        'QUALITY': quality_score
    }

    dominant = max(scores, key=scores.get)

    high_scores = [k for k, v in scores.items() if v > 0.60]

    if len(high_scores) >= 2:
        dominant = 'MIXED CRISIS'

    print('ROOT CAUSE ANALYSIS')
    print('-'*55)
    print(f'Shortage Score   : {shortage_score:.4f}')
    print(f'Disruption Score : {disruption_score:.4f}')
    print(f'Quality Score    : {quality_score:.4f}')
    print(f'\nDominant Cause   : {dominant}')
    print()

    # ==========================================================
    # SOLUTION ENGINE
    # ==========================================================
    recommendations = []

    if dominant == 'SHORTAGE':
        recommendations = [
            'Deploy emergency tanker water supply',
            'Activate reservoir conservation policy',
            'Enable groundwater reserve pumping',
            'Implement demand management / rationing'
        ]

    elif dominant == 'DISRUPTION':
        recommendations = [
            'Deploy pipeline repair team',
            'Restore network pressure',
            'Schedule urgent infrastructure maintenance',
            'Prioritize aging pipe replacement'
        ]

    elif dominant == 'QUALITY':
        recommendations = [
            'Start emergency purification treatment',
            'Apply pH balancing process',
            'Isolate contaminated supply lines',
            'Increase water quality monitoring'
        ]

    else:
        recommendations = [
            'Emergency tanker deployment',
            'Immediate leakage repair',
            'Pressure restoration',
            'Water purification treatment',
            'Groundwater reserve activation',
            'Emergency municipal response team activation'
        ]

    # ==========================================================
    # COUNTERFACTUAL IMPROVEMENT
    # ==========================================================
    improved = base.copy()

    if dominant in ['SHORTAGE', 'MIXED CRISIS']:
        improved['Rainfall(mm)'] *= 1.20
        improved['Flow_Rate(L/min)'] *= 1.25
        improved['Population_Demand'] *= 0.90

    if dominant in ['DISRUPTION', 'MIXED CRISIS']:
        improved['Leakage_History'] = max(0, improved['Leakage_History'] - 2)
        improved['Water_Pressure(psi)'] *= 1.20
        improved['Maintenance_History'] += 1

    if dominant in ['QUALITY', 'MIXED CRISIS']:
        improved['pH_Level'] = 7.0

    # recompute engineered
    improved['Demand_Supply_Ratio'] = (
        improved['Population_Demand'] /
        (improved['Flow_Rate(L/min)'] + 1)
    )

    improved['Pressure_Efficiency'] = (
        improved['Water_Pressure(psi)'] *
        improved['Pipe_Diameter(mm)']
    ) / (improved['Pipe_Age'] + 1)

    improved['Climate_Stress'] = (
        improved['Temperature(C)'] /
        (improved['Rainfall(mm)'] + 1)
    )

    improved['pH_Deviation'] = abs(improved['pH_Level'] - 7)

    new_prob = _risk_prob(best_model, scaler, X_cols, improved)

    improvement = (
        (current_prob - new_prob) /
        (current_prob + 1e-6)
    ) * 100

    improvement = max(0, improvement)

    # ==========================================================
    # COST / RESOURCE
    # ==========================================================
    severity = max(shortage_score, disruption_score, quality_score)

    priority = severity * 100

    teams = int(np.ceil(2 + severity * 4))
    cost = int(150000 + severity * 500000)

    # ==========================================================
    # DISPLAY
    # ==========================================================
    print('RECOMMENDED SOLUTION PLAN')
    print('-'*55)

    for i, rec in enumerate(recommendations, 1):
        print(f'{i}. {rec}')

    print()
    print(f'Priority Score            : {priority:.2f}/100')
    print(f'Resource Requirement      : {teams} operational teams')
    print(f'Estimated Implementation  : Rs. {cost:,}')
    print(f'Expected Risk Reduction   : {improvement:.2f}%')
    print()

    # =========================================================
    # ACTION 1: Leakage Repair
    # =========================================================
    if base['Leakage_History'] > 2:
        scenario = base.copy()
        scenario['Leakage_History'] = max(0, base['Leakage_History'] - 2)

        # recompute engineered
        scenario['Demand_Supply_Ratio'] = scenario['Population_Demand'] / (scenario['Flow_Rate(L/min)'] + 1)
        scenario['Pressure_Efficiency'] = (scenario['Water_Pressure(psi)'] * scenario['Pipe_Diameter(mm)']) / (scenario['Pipe_Age'] + 1)
        scenario['Climate_Stress'] = scenario['Temperature(C)'] / (scenario['Rainfall(mm)'] + 1)
        scenario['pH_Deviation'] = abs(scenario['pH_Level'] - 7)

        new_risk = _risk_prob(best_model, scaler, X_cols, scenario)

        # Teams = ceil((Leakage + Age_stress)/2)
        age_stress = _norm_series(df['Pipe_Age'], base['Pipe_Age'])
        teams = int(np.ceil((base['Leakage_History'] + age_stress) / 2))
        cost = 80000 * teams

        improvement = max(0.0, (current_risk - new_risk) / (current_risk + 1e-6) * 100)

        actions.append({
            'Action': 'Pipeline Leakage Repair',
            'Priority': priority_score,
            'Cost': int(cost),
            'Resource': f'{teams} repair teams',
            'Improvement': f'{improvement:.2f}%',
            'Level': 'CRITICAL' if priority_score > 80 else 'HIGH'
        })

    # =========================================================
    # ACTION 2: Pressure Restoration
    # =========================================================
    if base['Water_Pressure(psi)'] < df['Water_Pressure(psi)'].median():
        scenario = base.copy()
        scenario['Water_Pressure(psi)'] = df['Water_Pressure(psi)'].median()

        scenario['Pressure_Efficiency'] = (scenario['Water_Pressure(psi)'] * scenario['Pipe_Diameter(mm)']) / (scenario['Pipe_Age'] + 1)

        new_risk = _risk_prob(best_model, scaler, X_cols, scenario)

        improvement = max(0.0, (current_risk - new_risk) / (current_risk + 1e-6) * 100)

        cost = 120000

        actions.append({
            'Action': 'Pressure Restoration',
            'Priority': priority_score * 0.9,
            'Cost': cost,
            'Resource': '1 engineering team',
            'Improvement': f'{improvement:.2f}%',
            'Level': 'CRITICAL' if priority_score > 75 else 'HIGH'
        })

    # =========================================================
    # ACTION 3: Tanker Allocation
    # =========================================================
    demand_q75 = df['Population_Demand'].quantile(0.75)

    if base['Population_Demand'] > demand_q75:
        excess = base['Population_Demand'] - demand_q75

        # Tankers = ceil(excess / 100)
        tankers = int(np.ceil(excess / 100))
        cost = tankers * 25000

        scenario = base.copy()
        scenario['Flow_Rate(L/min)'] += tankers * 20

        scenario['Demand_Supply_Ratio'] = scenario['Population_Demand'] / (scenario['Flow_Rate(L/min)'] + 1)

        new_risk = _risk_prob(best_model, scaler, X_cols, scenario)

        improvement = max(0.0, (current_risk - new_risk) / (current_risk + 1e-6) * 100)

        actions.append({
            'Action': 'Tanker Water Allocation',
            'Priority': priority_score * 0.85,
            'Cost': int(cost),
            'Resource': f'{tankers} tankers',
            'Improvement': f'{improvement:.2f}%',
            'Level': 'HIGH'
        })

    # =========================================================
    # ACTION 4: Maintenance Improvement
    # =========================================================
    if base['Maintenance_History'] < df['Maintenance_History'].median():
        scenario = base.copy()
        scenario['Maintenance_History'] = df['Maintenance_History'].median()

        new_risk = _risk_prob(best_model, scaler, X_cols, scenario)

        improvement = max(0.0, (current_risk - new_risk) / (current_risk + 1e-6) * 100)

        cost = 100000

        actions.append({
            'Action': 'Infrastructure Maintenance Upgrade',
            'Priority': priority_score * 0.8,
            'Cost': cost,
            'Resource': 'Maintenance team',
            'Improvement': f'{improvement:.2f}%',
            'Level': 'MEDIUM'
        })

    # =========================================================
    # SORT & DISPLAY
    # =========================================================
    actions = sorted(actions, key=lambda x: x['Priority'], reverse=True)

    print(f'\nCurrent Predicted Risk Probability: {current_risk:.4f}\n')
    print('OPTIMIZED ACTION PLAN:\n')

    total_cost = 0

    for i, act in enumerate(actions, 1):
        total_cost += act['Cost']
        print(f'{i}. {act["Action"]}')
        print(f'   Level                : {act["Level"]}')
        print(f'   Priority Score       : {act["Priority"]:.2f}')
        print(f'   Estimated Cost       : Rs.{act["Cost"]:,}')
        print(f'   Resource Requirement : {act["Resource"]}')
        print(f'   Expected Improvement : {act["Improvement"]}')
        print()

    print('-'*90)
    print(f'Total Estimated Cost: Rs.{total_cost:,}')
    print('-'*90)
    print()


def custom_case_testing_engine(file_path, df_train, best_model, scaler, X):
    import pandas as pd
    import numpy as np

    print('\n' + '='*100)
    print('STEP 15 - CUSTOM TEST CASE INTELLIGENT EVALUATION ENGINE')
    print('='*100)

    # load
    test_df = pd.read_excel(file_path)

    # remove Case_ID for model
    case_ids = test_df['Case_ID'].copy()
    test_df_model = test_df.drop(columns=['Case_ID'])

    # ==========================================================
    # FEATURE ENGINEERING (same as training)
    # ==========================================================
    test_df_model['Demand_Supply_Ratio'] = (
        test_df_model['Population_Demand'] /
        (test_df_model['Flow_Rate(L/min)'] + 1)
    )

    test_df_model['Pressure_Efficiency'] = (
        test_df_model['Water_Pressure(psi)'] *
        test_df_model['Pipe_Diameter(mm)']
    ) / (test_df_model['Pipe_Age'] + 1)

    test_df_model['Climate_Stress'] = (
        test_df_model['Temperature(C)'] /
        (test_df_model['Rainfall(mm)'] + 1)
    )

    test_df_model['pH_Deviation'] = abs(
        test_df_model['pH_Level'] - 7
    )

    # column order match
    test_df_model = test_df_model[X.columns]

    # scale
    X_scaled = scaler.transform(test_df_model)

    # prediction
    probs = best_model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = []

    # normalization helper
    def norm(col, value):
        x = (
            (value - df_train[col].min()) /
            (df_train[col].max() - df_train[col].min() + 1e-6)
        )
        return float(np.clip(x, 0, 1))

    # ==========================================================
    # CASE-BY-CASE ANALYSIS
    # ==========================================================
    for i in range(len(test_df_model)):
        row = test_df_model.iloc[i]
        prob = probs[i]
        pred = preds[i]

        # stresses
        rainfall_stress = 1 - norm('Rainfall(mm)', row['Rainfall(mm)'])
        demand_stress = norm('Population_Demand', row['Population_Demand'])
        flow_stress = 1 - norm('Flow_Rate(L/min)', row['Flow_Rate(L/min)'])
        temp_stress = norm('Temperature(C)', row['Temperature(C)'])

        leakage_stress = norm('Leakage_History', row['Leakage_History'])
        pressure_stress = 1 - norm('Water_Pressure(psi)', row['Water_Pressure(psi)'])
        age_stress = norm('Pipe_Age', row['Pipe_Age'])
        maintenance_stress = 1 - norm('Maintenance_History', row['Maintenance_History'])

        quality_stress = min(abs(row['pH_Level'] - 7)/3, 1)

        shortage = (
            0.35*rainfall_stress +
            0.30*demand_stress +
            0.20*flow_stress +
            0.15*temp_stress
        )

        disruption = (
            0.35*leakage_stress +
            0.25*pressure_stress +
            0.20*age_stress +
            0.20*maintenance_stress
        )

        quality = quality_stress

        scores = {
            'SHORTAGE': shortage,
            'DISRUPTION': disruption,
            'QUALITY': quality
        }

        dominant = max(scores, key=scores.get)

        high = [k for k, v in scores.items() if v > 0.60]
        if len(high) >= 2:
            dominant = 'MIXED CRISIS'

        # solution mapping
        if pred == 0:
            solution = 'Routine monitoring'
        elif dominant == 'SHORTAGE':
            solution = 'Tanker + Reservoir conservation + Reserve pumping'
        elif dominant == 'DISRUPTION':
            solution = 'Leak repair + Pressure restoration + Maintenance'
        elif dominant == 'QUALITY':
            solution = 'Purification + pH balancing + Monitoring'
        else:
            solution = 'Integrated emergency municipal response'

        priority = max(shortage, disruption, quality) * 100
        teams = int(np.ceil(2 + max(shortage, disruption, quality)*4))
        cost = int(150000 + max(shortage, disruption, quality)*500000)

        results.append({
            'Case_ID': case_ids.iloc[i],
            'Risk_Probability': round(prob, 4),
            'Water_Risk': 'YES' if pred == 1 else 'NO',
            'Dominant_Cause': dominant if pred == 1 else 'NONE',
            'Priority_Score': round(priority, 2),
            'Operational_Teams': teams,
            'Estimated_Cost': cost,
            'Recommended_Solution': solution
        })

    result_df = pd.DataFrame(results)

    print('\nTop 10 Results:\n')
    print(result_df.head(10))

    # save report
    result_df.to_excel(
        'water_solution_report.xlsx',
        index=False
    )

    print('\nSaved: water_solution_report.xlsx\n')

    return result_df

#================Prediction Comparision========================#

def compare_models_by_case_id(file_path, case_id, tuned_models, scaler, X):
    print('\n' + '='*90)
    print(f'MODEL COMPARISON FOR {case_id}')
    print('='*90)

    # Load custom file
    df_case = pd.read_excel(file_path)

    # Find selected case
    row = df_case[df_case['Case_ID'] == case_id]

    if row.empty:
        print('Case_ID not found.')
        return

    row = row.iloc[0].copy()

    # Remove Case_ID
    features = row.drop('Case_ID').to_frame().T

    # Recompute engineered features
    features = recompute_engineered_features(features)

    # Match training columns
    features = features[X.columns]

    # Scale
    features_scaled = scaler.transform(features)

    predictions = []

    print('\nPredictions:')
    print('-'*70)

    for name, model in tuned_models.items():
        prob = model.predict_proba(features_scaled)[0][1]
        pred = model.predict(features_scaled)[0]

        label = 'YES' if pred == 1 else 'NO'
        predictions.append(label)

        print(f'{name:<20} : {label} ({prob*100:.2f}%)')

    # Consensus
    yes_count = predictions.count('YES')
    no_count = predictions.count('NO')

    print('\n' + '-'*70)

    if yes_count == 3:
        print('Consensus        : STRONG AGREEMENT (YES)')
        print('Final Prediction : WATER RISK = YES')

    elif no_count == 3:
        print('Consensus        : STRONG AGREEMENT (NO)')
        print('Final Prediction : WATER RISK = NO')

    else:
        print('Consensus        : MIXED PREDICTION')
        print(f'YES votes = {yes_count}, NO votes = {no_count}')

def plot_model_accuracy(results_df):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df['Model'],
        results_df['Test Accuracy'] * 100,
        edgecolor='black'
    )

    plt.title('Model-wise Test Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.grid(axis='y', alpha=0.3)

    for i, v in enumerate(results_df['Test Accuracy'] * 100):
        plt.text(i, v+0.2, f'{v:.2f}%', ha='center')

    plt.tight_layout()
    plt.show()

# =============================================================================
# TRAINING ACCURACY BAR CHART
# =============================================================================
def plot_training_accuracy(results_df):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))

    plt.bar(
        results_df['Model'],
        results_df['Train Accuracy'] * 100,
        edgecolor='black'
    )

    plt.title(
        'Model-wise Training Accuracy Comparison',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Training Accuracy (%)', fontsize=12)

    plt.ylim(80,101)
    plt.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=20)

    for i, v in enumerate(results_df['Train Accuracy'] * 100):
        plt.text(
            i,
            v + 0.2,
            f'{v:.2f}%',
            ha='center',
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    df = load_data()
    ##pd.set_option('display.max_columns', None)

    ##print('\nDATASET SAMPLE PREVIEW')
    ##print('='*100)
    ##print(df.head())
 

    df = clean_data(df)
    df = rebuild_target(df)
    df = engineer_features(df)
    plot_feature_histograms(df)

    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)

    trained_models = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    results_df = compare_models(trained_models, X_train_scaled, y_train, X_test_scaled, y_test)

    plot_training_accuracy(results_df)

    plot_model_accuracy(results_df)

    tuned_models = tune_top_models(results_df, X_train_scaled, y_train)

    best_model, final_model_name, cv_df = cross_validate_models(
        tuned_models,
        X_train_scaled,
        y_train
    )

    # RANDOM FOREST
    print('\n' + '='*80)
    print('RANDOM FOREST')
    print('='*80)

    rf_pred = tuned_models['Random Forest'].predict(X_test_scaled)

    print('Classification Report:')
    print(classification_report(
        y_test,
        rf_pred,
        target_names=['No Risk', 'Risk']
    ))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, rf_pred))


# GRADIENT BOOSTING
    print('\n' + '='*80)
    print('GRADIENT BOOSTING')
    print('='*80)

    gb_pred = tuned_models['Gradient Boosting'].predict(X_test_scaled)

    print('Classification Report:')
    print(classification_report(
        y_test,
        gb_pred,
        target_names=['No Risk', 'Risk']
    ))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, gb_pred))

    

    y_pred, cm, report = final_evaluation(best_model, X_test_scaled, y_test)

    plot_confusion_matrix_single(
    tuned_models['Random Forest'],
    'Random Forest',
    X_test_scaled,
    y_test
)

    plot_confusion_matrix_single(
    tuned_models['Gradient Boosting'],
    'Gradient Boosting',
    X_test_scaled,
    y_test
)

    plot_confusion_matrix_single(
    tuned_models['XGBoost'],
    'XGBoost',
    X_test_scaled,
    y_test
)

    plot_roc_curves(tuned_models, X_test_scaled, y_test)
    plot_precision_recall(tuned_models, X_test_scaled, y_test)
    plot_feature_importance(best_model, X)
    run_shap_analysis(best_model, X_train, X_test)
    simulate_scenarios(best_model, scaler, X)
    diagnose_and_optimize(df, best_model, scaler, X)
    custom_case_testing_engine(
    'custom_water_test_cases_50_with_soil.xlsx',
    df,
    best_model,
    scaler,
    X
)
    
    compare_models_by_case_id(
    'custom_water_test_cases_50_with_soil.xlsx',
    'Case_17',
    tuned_models,
    scaler,
    X
)

