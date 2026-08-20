# %% [markdown]
# # BenignIDS — Enhanced Training Pipeline with Analysis & Reporting
# 
# **New Features:**
# - Model comparison tables
# - Bayesian Optimization landscape visualization
# - OpenAI-powered executive summary reports

# %%
# =====================================================
# Additional Imports for Enhanced Features
# =====================================================
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

# =====================================================
# Configuration Class
# =====================================================
@dataclass
class Config:
    """Configuration class for the BenignIDS pipeline"""
    DATA_PATH: str = "archive/Payload_data_UNSW.csv"
    OUT_ROOT: Path = Path("out")
    STAGE_ROOT: Path = Path("staging")
    TARGET_COL: str = "label"
    RANDOM_STATE: int = 42
    BO_N_CALLS_MAX: int = 50
    BO_N_RANDOM_STARTS: int = 10
    
    def __post_init__(self):
        self.OUT_ROOT = Path(self.OUT_ROOT)
        self.STAGE_ROOT = Path(self.STAGE_ROOT)
        # Create directories if they don't exist
        self.OUT_ROOT.mkdir(exist_ok=True)
# Initialize global config
config = Config()

# =====================================================
# Utility Functions
# =====================================================
class Utils:
    @staticmethod
    def safe_ap_score(y_true, y_pred):
        """Safely calculate average precision score"""
        try:
            return average_precision_score(y_true, y_pred)
        except Exception:
            return 0.0

utils = Utils()

# =====================================================
# Data Processing Classes (Stub implementations)
# =====================================================
class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
    
    def load_and_prepare_data(self):
        """Load and prepare data - stub implementation"""
        try:
            df = pd.read_csv(self.config.DATA_PATH)
            y = df[self.config.TARGET_COL] if self.config.TARGET_COL in df.columns else pd.Series([0, 1] * (len(df)//2))
            # Ensure y is numeric
            y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
            return df, y
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for demonstration
            n_samples = 1000
            df = pd.DataFrame({
                'feature_1': np.random.randn(n_samples),
                'feature_2': np.random.randn(n_samples),
                'feature_3': np.random.randn(n_samples),
                'label': np.random.choice([0, 1], n_samples)
            })
            y = df['label'].astype(int)
            return df, y
    
    def create_splits(self, df, y):
        """Create train/val/test splits"""
        X = df.drop(columns=[self.config.TARGET_COL] if self.config.TARGET_COL in df.columns else [])
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=self.config.RANDOM_STATE)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.config.RANDOM_STATE)
        
        return {
            "X_train": X_train,
            "X_val": X_val, 
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }
    
    def create_payload_features(self, splits):
        """Create payload features - stub implementation"""
        # Use numeric features as payload features
        X_train_numeric = splits["X_train"].select_dtypes(include=[np.number])
        X_val_numeric = splits["X_val"].select_dtypes(include=[np.number])
        X_test_numeric = splits["X_test"].select_dtypes(include=[np.number])
        
        return {
            "X_train_payload": X_train_numeric,
            "X_val_payload": X_val_numeric,
            "X_test_payload": X_test_numeric
        }

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def select_features(self, X, y, k=10):
        """Select top k features"""
        if X.empty:
            return []
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()

class ModelTrainer:
    def __init__(self, config: Config):
    def train_lightgbm_baseline(self, X_train, X_val, y_train, y_val):
        """Train baseline LightGBM model"""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required for training")
        
        # Ensure labels are numeric
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
        y_val = pd.to_numeric(y_val, errors='coerce').fillna(0).astype(int)
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'seed': self.config.RANDOM_STATE
        }
        
        model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=100,
                         callbacks=[lgb.early_stopping(20, verbose=False)])
        
        return {"model": model}
        
        return {"model": model}
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model and return metrics"""
        try:
            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            return {
                'ap': average_precision_score(y_test, y_pred_proba),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'n_samples': len(y_test)
            }
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return {
                'ap': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 
                'precision': 0.0, 'recall': 0.0, 'n_samples': len(y_test)
            }

# Initialize global config
config = Config()

# =====================================================
# Enhanced Model Comparison & Analysis
# =====================================================
class ModelAnalyzer:
    """Advanced model analysis and comparison"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results_data = []
        
    def add_model_result(self, model_name: str, metrics: Dict, model_type: str = "ML"):
        """Add model results for comparison"""
        self.results_data.append({
            'Model': model_name,
    def generate_comparison_table(self) -> Tuple[str, pd.DataFrame]:
        """Generate formatted comparison table"""
        if not self.results_data:
            empty_df = pd.DataFrame()
            return "No model results available", empty_df
            
        df = pd.DataFrame(self.results_data)
        
        # Sort by AP Score descending
        df = df.sort_values('AP Score', ascending=False)
        
        # Format numeric columns
        numeric_cols = ['AP Score', 'ROC AUC', 'F1 Score', 'Precision', 'Recall']
        for col in numeric_cols:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        # Create table
        if HAS_TABULATE:
            table_str = tabulate(df.values, headers=df.columns, tablefmt='grid')
        else:
            table_str = df.to_string()
        
        # Add ranking
        df_ranked = df.copy()
        df_ranked.insert(0, 'Rank', range(1, len(df_ranked) + 1))
        
        return table_str, df_ranked
        
        # Create table
        table_str = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        
        # Add ranking
        df_ranked = df.copy()
    def create_performance_heatmap(self) -> Optional['plt.Figure']:
        """Create performance heatmap"""
        if not self.results_data or not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
            
        df = pd.DataFrame(self.results_data)
        
        # Select numeric columns for heatmap
        numeric_cols = ['AP Score', 'ROC AUC', 'F1 Score', 'Precision', 'Recall']
        heatmap_data = df[['Model'] + numeric_cols].set_index('Model')
        
        # Convert to numeric
        for col in numeric_cols:
            heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', 
                   center=0.5, fmt='.4f', ax=ax)
        plt.title('Model Performance Comparison Heatmap')
        plt.tight_layout()
        
        return fig
        heatmap_data = df[['Model'] + numeric_cols].set_index('Model')
        
        # Convert to numeric
        for col in numeric_cols:
            heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', 
                   center=0.5, fmt='.4f', ax=ax)
        plt.title('Model Performance Comparison Heatmap')
        plt.tight_layout()
        
        return fig
    def optimize_lightgbm(self, X_train, X_val, y_train, y_val) -> Dict:
        """Optimize LightGBM hyperparameters"""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required for optimization")
        
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        # Ensure labels are numeric
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
        y_val = pd.to_numeric(y_val, errors='coerce').fillna(0).astype(int)
        
        # Define search space
        space = [
            Real(0.01, 0.3, name='learning_rate'),
            Integer(10, 150, name='num_leaves'),
            Real(0.5, 1.0, name='feature_fraction'),
            Real(0.5, 1.0, name='bagging_fraction'),
            Integer(50, 500, name='n_estimators')
        ]
        
        @use_named_args(space)
        def objective(**params):
            """Objective function for optimization"""
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            lgb_params = {
                'objective': 'binary',
                'metric': 'None',  # We'll use custom eval
                'verbosity': -1,
                'seed': self.config.RANDOM_STATE,
                **params
            }
            
            # Extract n_estimators separately
            n_est = params.pop('n_estimators', 100)
            
            model = lgb.train(
                lgb_params, dtrain,
                valid_sets=[dval],
                num_boost_round=n_est,
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            # Predict and calculate AP score
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            ap_score = utils.safe_ap_score(y_val, y_pred)
            
            # Store result
            result = {**params, 'n_estimators': n_est, 'ap_score': ap_score}
            self.optimization_results.append(result)
            self.objective_values.append(-ap_score)  # Minimize negative AP
            
            return -ap_score  # Return negative for minimization
        
        print("Starting Bayesian Optimization...")
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.config.BO_N_CALLS_MAX,
            n_random_starts=self.config.BO_N_RANDOM_STARTS,
            random_state=self.config.RANDOM_STATE,
            acq_func='EI'
        )
        
        self.best_params = dict(zip([dim.name for dim in space], result.x))
        
        return {
            'best_params': self.best_params,
    def create_bo_landscape(self) -> Optional['plt.Figure']:
        """Create Bayesian Optimization landscape visualization"""
        if not self.optimization_results or not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Convergence plot
        ax1 = axes[0, 0]
        iterations = range(1, len(self.objective_values) + 1)
        cumulative_best = np.minimum.accumulate(self.objective_values)
        
        ax1.plot(iterations, [-x for x in self.objective_values], 'b.', alpha=0.6, label='Observations')
        ax1.plot(iterations, [-x for x in cumulative_best], 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('AP Score')
        ax1.set_title('Bayesian Optimization Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter importance
        ax2 = axes[0, 1]
        df_results = pd.DataFrame(self.optimization_results)
        param_cols = ['learning_rate', 'num_leaves', 'feature_fraction', 'bagging_fraction', 'n_estimators']
        
        # Calculate correlation with AP score
        correlations = []
        for col in param_cols:
            if col in df_results.columns:
                corr = df_results[col].corr(df_results['ap_score'])
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        
        ax2.barh(param_cols, correlations)
        ax2.set_xlabel('Absolute Correlation with AP Score')
        ax2.set_title('Parameter Importance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate vs AP score
        ax3 = axes[1, 0]
        if 'learning_rate' in df_results.columns:
            scatter = ax3.scatter(df_results['learning_rate'], df_results['ap_score'], 
                                c=range(len(df_results)), cmap='viridis', alpha=0.7)
            ax3.set_xlabel('Learning Rate')
            ax3.set_ylabel('AP Score')
            ax3.set_title('Learning Rate vs Performance')
            plt.colorbar(scatter, ax=ax3, label='Iteration')
        
        # 4. Num leaves vs AP score
        ax4 = axes[1, 1]
        if 'num_leaves' in df_results.columns:
            scatter = ax4.scatter(df_results['num_leaves'], df_results['ap_score'], 
                                c=range(len(df_results)), cmap='viridis', alpha=0.7)
            ax4.set_xlabel('Number of Leaves')
            ax4.set_ylabel('AP Score')
            ax4.set_title('Number of Leaves vs Performance')
            plt.colorbar(scatter, ax=ax4, label='Iteration')
        
        plt.tight_layout()
        return fig
                                c=range(len(df_results)), cmap='viridis', alpha=0.7)
            ax3.set_xlabel('Learning Rate')
            ax3.set_ylabel('AP Score')
            ax3.set_title('Learning Rate vs Performance')
            plt.colorbar(scatter, ax=ax3, label='Iteration')
        
        # 4. Num leaves vs AP score
    def __init__(self, config: Config, api_key: Optional[str] = None):
        self.config = config
        if api_key and HAS_OPENAI:
            openai.api_key = api_key
        
    def generate_executive_summary(self, results_data: Dict, model_comparison: pd.DataFrame, 
                                 bo_results: Optional[Dict] = None) -> str:
            plt.colorbar(scatter, ax=ax4, label='Iteration')
        
        plt.tight_layout()
        return fig

# =====================================================
# OpenAI Report Generator
# =====================================================
class ReportGenerator:
    """Generate executive summary reports using OpenAI API"""
    
    def __init__(self, config: Config, api_key: str = None):
        self.config = config
        if api_key:
            openai.api_key = api_key
        
    def generate_executive_summary(self, results_data: Dict, model_comparison: pd.DataFrame, 
                                 bo_results: Dict = None) -> str:
        """Generate executive summary using OpenAI"""
        
        # Prepare data summary
        best_model = model_comparison.iloc[0] if not model_comparison.empty else {}
        
        data_summary = {
            "dataset_info": {
                "total_samples": results_data.get("config", {}).get("train_size", 0) + 
                               results_data.get("config", {}).get("val_size", 0) + 
                               results_data.get("config", {}).get("test_size", 0),
                "train_samples": results_data.get("config", {}).get("train_size", 0),
                "validation_samples": results_data.get("config", {}).get("val_size", 0),
                "test_samples": results_data.get("config", {}).get("test_size", 0)
            },
            "best_model": {
                "name": best_model.get("Model", "Unknown"),
                "ap_score": best_model.get("AP Score", "N/A"),
                "roc_auc": best_model.get("ROC AUC", "N/A"),
                "f1_score": best_model.get("F1 Score", "N/A")
            },
            "models_tested": len(model_comparison),
            "optimization_results": bo_results
        }
        
        prompt = f"""
        As a senior ML engineer, write an executive summary report for a cybersecurity intrusion detection system (BenignIDS) project.

        Project Data:
        - Total dataset size: {data_summary['dataset_info']['total_samples']:,} samples
        - Training samples: {data_summary['dataset_info']['train_samples']:,}
        try:
            if HAS_OPENAI:
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a senior ML engineer specializing in cybersecurity."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                
                return response.choices[0].message.content
            else:
                raise ImportError("OpenAI library not available")
        4. Recommendations for production deployment
        5. Next Steps

        Keep it professional, concise, and suitable for technical stakeholders.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior ML engineer specializing in cybersecurity."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"""
            # BenignIDS Executive Summary Report
def run_enhanced_pipeline(openai_api_key: Optional[str] = None):
            **Note: OpenAI API not available - Generated template report**
            
            ## Executive Summary
            Successfully trained and evaluated {data_summary['models_tested']} machine learning models for intrusion detection on a dataset of {data_summary['dataset_info']['total_samples']:,} samples. The best performing model achieved an AP Score of {data_summary['best_model']['ap_score']}.
            
            ## Key Findings
            - Best model: {data_summary['best_model']['name']}
            - Peak performance: AP Score {data_summary['best_model']['ap_score']}, ROC AUC {data_summary['best_model']['roc_auc']}
            - Dataset split: {data_summary['dataset_info']['train_samples']:,} train, {data_summary['dataset_info']['validation_samples']:,} validation, {data_summary['dataset_info']['test_samples']:,} test
            
            ## Recommendations
            - Proceed with {data_summary['best_model']['name']} for production deployment
            - Implement continuous monitoring and retraining pipeline
            - Consider ensemble methods for improved robustness
            
            Error generating detailed report: {str(e)}
            """

# =====================================================
# Enhanced Pipeline with Analysis
# =====================================================
def run_enhanced_pipeline(openai_api_key: str = None):
    """Run enhanced pipeline with analysis and reporting"""
    print("=== Starting Enhanced BenignIDS Training Pipeline ===")
    
    # Initialize components
    processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    analyzer = ModelAnalyzer(config)
    bo_optimizer = BayesianOptimizer(config)
    report_generator = ReportGenerator(config, openai_api_key)
    
    # Run basic pipeline steps (1-4 same as before)
    print("\n1-4. Running basic pipeline steps...")
    df, y = processor.load_and_prepare_data()
    splits = processor.create_splits(df, y)
    
    try:
        payload_features = processor.create_payload_features(splits)
        X_train_payload = payload_features["X_train_payload"]
        X_val_payload = payload_features["X_val_payload"]
        X_test_payload = payload_features["X_test_payload"]
    except Exception as e:
        print(f"Warning: Could not create payload features: {e}")
        X_train_payload = X_val_payload = X_test_payload = None
    
    # Feature selection
    X_train_struct = splits["X_train"].select_dtypes(include=[np.number])
    if not X_train_struct.empty:
        selected_features = feature_engineer.select_features(X_train_struct, splits["y_train"])
        X_train_selected = X_train_struct[selected_features]
        X_val_selected = splits["X_val"][selected_features]
    # 6. Bayesian Optimization
    print("\n6. Running Bayesian Optimization...")
    bo_results = None
    if X_train_payload is not None and HAS_SKOPT:
        try:
            bo_results = bo_optimizer.optimize_lightgbm(
                X_train_payload, X_val_payload,
                splits["y_train"], splits["y_val"]
            )
            
            # Train final model with best parameters
            y_train_numeric = pd.to_numeric(splits["y_train"], errors='coerce').fillna(0).astype(int)
            y_test_numeric = pd.to_numeric(splits["y_test"], errors='coerce').fillna(0).astype(int)
            
            dtrain = lgb.Dataset(X_train_payload, label=y_train_numeric)
            dtest = lgb.Dataset(X_test_payload, label=y_test_numeric, reference=dtrain)
            
            best_params = bo_results['best_params'].copy()
            n_est = best_params.pop('n_estimators', 100)
            
            final_params = {
                'objective': 'binary',
                'metric': 'None',
                'verbosity': -1,
                'seed': config.RANDOM_STATE,
                **best_params
            }
            
            optimized_model = lgb.train(
                final_params, dtrain,
                num_boost_round=n_est,
                valid_sets=[dtest],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            # Evaluate optimized model
            optimized_metrics = trainer.evaluate_model(
                "LightGBM_Payload_Optimized", optimized_model,
                X_test_payload, splits["y_test"]
            )
            analyzer.add_model_result("LightGBM Payload (Optimized)", optimized_metrics, "ML")
            
            results["lgbm_optimized"] = {
                "model": optimized_model,
                "params": bo_results['best_params'],
                "val_ap": bo_results['best_score']
            }
        except Exception as e:
            print(f"Bayesian optimization failed: {e}")
            bo_results = None
    else:
        print("Skipping Bayesian Optimization (dependencies not available)")
            **best_params
        }
        
        optimized_model = lgb.train(
            final_params, dtrain,
            num_boost_round=n_est,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        
    # Create reports directory
    reports_dir = config.OUT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Performance heatmap
    heatmap_fig = analyzer.create_performance_heatmap()
    if heatmap_fig and HAS_MATPLOTLIB:
        heatmap_path = reports_dir / "performance_heatmap.png"
        heatmap_fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(heatmap_fig)
    
    # BO landscape
    if hasattr(bo_optimizer, 'optimization_results') and bo_optimizer.optimization_results and HAS_MATPLOTLIB:
        bo_fig = bo_optimizer.create_bo_landscape()
        if bo_fig:
            bo_path = reports_dir / "bo_landscape.png"
    # 9. Generate Executive Summary
    print("\n9. Generating executive summary report...")
    exec_summary = report_generator.generate_executive_summary(
        results, comparison_df, bo_results
    )
    
    # Save executive summary
    summary_path = reports_dir / "executive_summary.md"
    summary_path.write_text(exec_summary)
    # Best model recommendation
    best_model = analyzer.get_best_model()
    print(f"\n=== BEST MODEL RECOMMENDATION ===")
    print(f"Recommended Model: {best_model.get('Model', 'Unknown')}")
    print(f"AP Score: {best_model.get('AP Score', 'N/A')}")
    print(f"F1 Score: {best_model.get('F1 Score', 'N/A')}")
    
    # Create visualizations
    print("\n8. Creating visualizations...")
    
    # Performance heatmap
    heatmap_fig = analyzer.create_performance_heatmap()
    if heatmap_fig:
        heatmap_path = config.OUT_ROOT / "reports" / "performance_heatmap.png"
        "model_comparison": comparison_df.to_dict('records') if not comparison_df.empty else [],
        "best_model": best_model,
        "bayesian_optimization": bo_results,
    # BO landscape
    if hasattr(bo_optimizer, 'optimization_results') and bo_optimizer.optimization_results:
        bo_fig = bo_optimizer.create_bo_landscape()
        if bo_fig:
            bo_path = config.OUT_ROOT / "reports" / "bo_landscape.png"
            bo_fig.savefig(bo_path, dpi=300, bbox_inches='tight')
            plt.close(bo_fig)
            print(f"BO Landscape saved to: {bo_path}")
    
    # 9. Generate Executive Summary
    print("\n9. Generating executive summary report...")
    exec_summary = report_generator.generate_executive_summary(
        results, comparison_df, bo_results if 'bo_results' in locals() else None
    )
    
    # Save executive summary
    summary_path = config.OUT_ROOT / "reports" / "executive_summary.md"
    summary_path.write_text(exec_summary)
    
    # 10. Save comprehensive results
    print("\n10. Saving comprehensive results...")
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_config": {
            "random_state": config.RANDOM_STATE,
            "data_path": str(config.DATA_PATH),
            "dataset_sizes": {
                "train": len(splits["y_train"]),
                "val": len(splits["y_val"]),
                "test": len(splits["y_test"])
            }
        },
        "model_comparison": comparison_df.to_dict('records') if not comparison_df.empty else [],
        "best_model": best_model,
        "bayesian_optimization": bo_results if 'bo_results' in locals() else None,
        "files_generated": {
            "executive_summary": str(summary_path),
            "performance_heatmap": str(config.OUT_ROOT / "reports" / "performance_heatmap.png"),
            "bo_landscape": str(config.OUT_ROOT / "reports" / "bo_landscape.png")
        }
    }
    
    results_path = config.OUT_ROOT / "comprehensive_results.json"
    results_path.write_text(json.dumps(final_results, indent=2))
    
    print(f"\n=== Enhanced Pipeline Complete ===")
    print(f"Results saved to: {config.OUT_ROOT}")
    print(f"Executive summary: {summary_path}")
    print(f"Comprehensive results: {results_path}")
    print(f"\nBest Model: {best_model.get('Model', 'Unknown')} (AP: {best_model.get('AP Score', 'N/A')})")
    
    return final_results

# =====================================================
# Execute Enhanced Pipeline
# =====================================================
if __name__ == "__main__":
    # Set up global variables
    globals().update({
        "OUT_ROOT": "out",
        "STAGE_ROOT": "staging", 
        "DATA_PATH": "archive/Payload_data_UNSW.csv",
        "TARGET_COL": "label",
        "RANDOM_STATE": 42
    })
    
    # Optional: Set OpenAI API key
    OPENAI_API_KEY = None  # Replace with your API key or set as environment variable
    
    try:
        results = run_enhanced_pipeline(OPENAI_API_KEY)
        print("\nEnhanced pipeline executed successfully!")
        print("\nGenerated files:")
        for file_type, path in results.get("files_generated", {}).items():
            print(f"- {file_type}: {path}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
