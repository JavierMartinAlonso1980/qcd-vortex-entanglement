"""
Belle II Tau Pair Entanglement Classifier
==========================================

Machine learning-based classification of τ⁺τ⁻ events into:
- Entangled (quantum correlations dominant)
- Thermal (classical thermal noise dominant)
- Mixed (intermediate regime)

Implements algorithm from file:3: Fermionic Bulk-Boundary Adaptation
Uses features from file:2: Belle II Tau Pair Toy MC

Classification pipeline:
1. Feature extraction (kinematic + helicity variables)
2. Entanglement witness calculation
3. ML classification (XGBoost/Random Forest)
4. Post-processing and validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import warnings

try:
    import uproot  # ROOT file I/O
    UPROOT_AVAILABLE = True
except ImportError:
    UPROOT_AVAILABLE = False
    warnings.warn("uproot not available, ROOT file reading disabled")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, ML classification disabled")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class EntanglementFeatureExtractor:
    """
    Extract entanglement-sensitive features from Belle II τ⁺τ⁻ events.
    
    Features based on file:2 mainbelle.txt and file:3 theoretical model:
    - Kinematic: p_T, η, φ, missing E_T
    - Angular: cos(θ_τ+), cos(θ_τ-), Δφ
    - Helicity: h_+, h_-, correlation C_hh
    - Spin density matrix elements
    - Concurrence estimator (file:3 Eq. 17)
    """
    
    def __init__(self, cms_energy: float = 10.58):
        """
        Args:
            cms_energy: Center-of-mass energy in GeV (Υ(4S) mass)
        """
        self.E_cms = cms_energy
        
        # Feature names
        self.feature_names = [
            # Kinematic
            'tau_plus_pT', 'tau_plus_eta', 'tau_plus_phi',
            'tau_minus_pT', 'tau_minus_eta', 'tau_minus_phi',
            'missing_ET', 'missing_phi',
            'invariant_mass',
            
            # Angular correlations
            'cos_theta_plus', 'cos_theta_minus',
            'delta_phi',
            'opening_angle',
            'acoplanarity',
            
            # Helicity (if available)
            'helicity_plus', 'helicity_minus',
            'helicity_correlation',
            
            # Spin density matrix
            'rho_00', 'rho_11', 'rho_01_real', 'rho_01_imag',
            
            # Entanglement estimators
            'concurrence_estimate',
            'log_negativity_estimate',
            'bell_parameter_S',
        ]
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from Belle II event DataFrame.
        
        Expected columns (from basf2 steering file output):
        - tau_plus_px, tau_plus_py, tau_plus_pz, tau_plus_E
        - tau_minus_px, tau_minus_py, tau_minus_pz, tau_minus_E
        - (optional) helicity_plus, helicity_minus
        
        Args:
            df: Input DataFrame with raw event data
        
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # === Kinematic features ===
        
        # Transverse momentum
        features['tau_plus_pT'] = np.sqrt(df['tau_plus_px']**2 + df['tau_plus_py']**2)
        features['tau_minus_pT'] = np.sqrt(df['tau_minus_px']**2 + df['tau_minus_py']**2)
        
        # Pseudorapidity η = -ln(tan(θ/2))
        tau_plus_p = np.sqrt(df['tau_plus_px']**2 + df['tau_plus_py']**2 + df['tau_plus_pz']**2)
        tau_minus_p = np.sqrt(df['tau_minus_px']**2 + df['tau_minus_py']**2 + df['tau_minus_pz']**2)
        
        features['tau_plus_eta'] = self._pseudorapidity(df['tau_plus_pz'], tau_plus_p)
        features['tau_minus_eta'] = self._pseudorapidity(df['tau_minus_pz'], tau_minus_p)
        
        # Azimuthal angle φ
        features['tau_plus_phi'] = np.arctan2(df['tau_plus_py'], df['tau_plus_px'])
        features['tau_minus_phi'] = np.arctan2(df['tau_minus_py'], df['tau_minus_px'])
        
        # Missing transverse energy (neutrinos)
        total_px = df['tau_plus_px'] + df['tau_minus_px']
        total_py = df['tau_plus_py'] + df['tau_minus_py']
        features['missing_ET'] = np.sqrt(total_px**2 + total_py**2)
        features['missing_phi'] = np.arctan2(total_py, total_px)
        
        # Invariant mass
        total_E = df['tau_plus_E'] + df['tau_minus_E']
        total_pz = df['tau_plus_pz'] + df['tau_minus_pz']
        features['invariant_mass'] = np.sqrt(total_E**2 - total_px**2 - total_py**2 - total_pz**2)
        
        # === Angular correlations ===
        
        # Polar angles in CMS frame
        features['cos_theta_plus'] = df['tau_plus_pz'] / tau_plus_p
        features['cos_theta_minus'] = df['tau_minus_pz'] / tau_minus_p
        
        # Azimuthal separation
        features['delta_phi'] = self._delta_phi(features['tau_plus_phi'], features['tau_minus_phi'])
        
        # Opening angle
        features['opening_angle'] = self._opening_angle(
            df[['tau_plus_px', 'tau_plus_py', 'tau_plus_pz']].values,
            df[['tau_minus_px', 'tau_minus_py', 'tau_minus_pz']].values
        )
        
        # Acoplanarity (deviation from back-to-back in transverse plane)
        features['acoplanarity'] = np.abs(features['delta_phi'] - np.pi)
        
        # === Helicity features (if available) ===
        
        if 'helicity_plus' in df.columns and 'helicity_minus' in df.columns:
            features['helicity_plus'] = df['helicity_plus']
            features['helicity_minus'] = df['helicity_minus']
            features['helicity_correlation'] = df['helicity_plus'] * df['helicity_minus']
        else:
            # Estimate from kinematics
            features['helicity_plus'] = self._estimate_helicity(
                df[['tau_plus_px', 'tau_plus_py', 'tau_plus_pz']].values
            )
            features['helicity_minus'] = self._estimate_helicity(
                df[['tau_minus_px', 'tau_minus_py', 'tau_minus_pz']].values
            )
            features['helicity_correlation'] = features['helicity_plus'] * features['helicity_minus']
        
        # === Spin density matrix (file:2 equations) ===
        
        rho_elements = self._spin_density_matrix(
            features['helicity_plus'].values,
            features['helicity_minus'].values,
            features['cos_theta_plus'].values,
            features['cos_theta_minus'].values
        )
        
        features['rho_00'] = rho_elements[:, 0]
        features['rho_11'] = rho_elements[:, 1]
        features['rho_01_real'] = rho_elements[:, 2]
        features['rho_01_imag'] = rho_elements[:, 3]
        
        # === Entanglement estimators ===
        
        # Concurrence estimate (file:3 Eq. 17)
        features['concurrence_estimate'] = self._estimate_concurrence(
            features['helicity_plus'].values,
            features['helicity_minus'].values,
            features['cos_theta_plus'].values,
            features['cos_theta_minus'].values
        )
        
        # Log-negativity proxy (from spin correlations)
        features['log_negativity_estimate'] = self._estimate_log_negativity(
            features['concurrence_estimate'].values
        )
        
        # Bell parameter S (CHSH inequality)
        features['bell_parameter_S'] = self._bell_parameter(
            features['helicity_correlation'].values,
            features['delta_phi'].values
        )
        
        return features
    
    def extract_from_root_file(self, root_file: str, tree_name: str = 'tau_ntuple') -> pd.DataFrame:
        """
        Extract features directly from Belle II ROOT file.
        
        Args:
            root_file: Path to ROOT file
            tree_name: Name of TTree
        
        Returns:
            DataFrame with features
        """
        if not UPROOT_AVAILABLE:
            raise ImportError("uproot is required for ROOT file reading")
        
        # Load ROOT file
        with uproot.open(root_file) as f:
            tree = f[tree_name]
            
            # Extract branches
            df = tree.arrays([
                'tau_plus_px', 'tau_plus_py', 'tau_plus_pz', 'tau_plus_E',
                'tau_minus_px', 'tau_minus_py', 'tau_minus_pz', 'tau_minus_E',
            ], library='pd')
        
        # Extract features
        return self.extract_from_dataframe(df)
    
    @staticmethod
    def _pseudorapidity(pz: np.ndarray, p: np.ndarray) -> np.ndarray:
        """η = -ln(tan(θ/2)) = arctanh(pz/p)"""
        cos_theta = pz / (p + 1e-10)
        cos_theta = np.clip(cos_theta, -0.9999, 0.9999)
        return np.arctanh(cos_theta)
    
    @staticmethod
    def _delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        """Azimuthal separation with wraparound."""
        dphi = phi1 - phi2
        dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        return np.abs(dphi)
    
    @staticmethod
    def _opening_angle(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """3D opening angle between momentum vectors."""
        dot_product = np.sum(p1 * p2, axis=1)
        mag1 = np.linalg.norm(p1, axis=1)
        mag2 = np.linalg.norm(p2, axis=1)
        
        cos_angle = dot_product / (mag1 * mag2 + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        return np.arccos(cos_angle)
    
    @staticmethod
    def _estimate_helicity(momentum: np.ndarray) -> np.ndarray:
        """
        Estimate helicity from momentum direction.
        
        Simplified: h ≈ sign(p_z) for forward/backward
        Real analysis requires decay product analysis.
        """
        return np.sign(momentum[:, 2])
    
    def _spin_density_matrix(self, h_plus: np.ndarray, h_minus: np.ndarray,
                            cos_theta_plus: np.ndarray, cos_theta_minus: np.ndarray) -> np.ndarray:
        """
        Construct spin density matrix elements (file:2).
        
        For spin-1/2 system: ρ is 4×4 matrix in {|++⟩, |+-⟩, |-+⟩, |--⟩} basis
        
        Returns:
            Array of shape (N_events, 4) with [ρ_00, ρ_11, Re(ρ_01), Im(ρ_01)]
        """
        N = len(h_plus)
        rho = np.zeros((N, 4))
        
        # Diagonal elements (populations)
        P_plus_plus = 0.25 * (1 + cos_theta_plus) * (1 + cos_theta_minus)
        P_minus_minus = 0.25 * (1 - cos_theta_plus) * (1 - cos_theta_minus)
        
        rho[:, 0] = P_plus_plus
        rho[:, 1] = P_minus_minus
        
        # Off-diagonal (coherences) - simplified model
        rho[:, 2] = 0.25 * np.sqrt(P_plus_plus * P_minus_minus)  # Real part
        rho[:, 3] = 0.0  # Imaginary part (requires full analysis)
        
        return rho
    
    def _estimate_concurrence(self, h_plus: np.ndarray, h_minus: np.ndarray,
                             cos_theta_plus: np.ndarray, cos_theta_minus: np.ndarray) -> np.ndarray:
        """
        Estimate concurrence from kinematic observables (file:3).
        
        Simplified proxy: C ≈ |correlation coefficient| × kinematic factor
        """
        # Spin-spin correlation
        C_hh = h_plus * h_minus
        
        # Angular correlation factor
        angular_factor = np.abs(cos_theta_plus * cos_theta_minus)
        
        # Concurrence estimate (normalized to [0, 1])
        C_est = np.abs(C_hh * angular_factor)
        
        return np.clip(C_est, 0, 1)
    
    @staticmethod
    def _estimate_log_negativity(concurrence: np.ndarray) -> np.ndarray:
        """
        Estimate log-negativity from concurrence.
        
        Approximate relation: E_N ≈ log₂(1 + C)
        """
        return np.log2(1 + concurrence)
    
    @staticmethod
    def _bell_parameter(helicity_corr: np.ndarray, delta_phi: np.ndarray) -> np.ndarray:
        """
        Estimate CHSH Bell parameter S.
        
        S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
        
        Simplified: S ≈ 2√2 × |⟨h+ h-⟩| × angular_factor
        """
        angular_factor = np.abs(np.cos(delta_phi))
        
        S = 2 * np.sqrt(2) * np.abs(helicity_corr) * angular_factor
        
        return S


class KinematicSelector:
    """
    Apply kinematic selection cuts to Belle II events.
    
    Implements selection criteria from file:2 and Belle II standard analysis.
    """
    
    def __init__(self, selection_config: Optional[Dict] = None):
        """
        Args:
            selection_config: Custom selection criteria (default: standard Belle II)
        """
        if selection_config is None:
            from . import EVENT_SELECTION
            self.config = EVENT_SELECTION
        else:
            self.config = selection_config
    
    def apply_cuts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply selection cuts and return passing events.
        
        Args:
            df: Input DataFrame with features
        
        Returns:
            (filtered_df, mask_passed)
        """
        mask = np.ones(len(df), dtype=bool)
        
        # Momentum cuts
        if 'tau_plus_pT' in df.columns:
            mask &= (df['tau_plus_pT'] > self.config['min_tau_momentum'])
            mask &= (df['tau_plus_pT'] < self.config['max_tau_momentum'])
            mask &= (df['tau_minus_pT'] > self.config['min_tau_momentum'])
            mask &= (df['tau_minus_pT'] < self.config['max_tau_momentum'])
        
        # Invariant mass window
        if 'invariant_mass' in df.columns:
            E_min, E_max = self.config['cms_energy_window']
            mask &= (df['invariant_mass'] > E_min)
            mask &= (df['invariant_mass'] < E_max)
        
        # Missing mass cut (reduce background)
        if 'missing_ET' in df.columns:
            mask &= (df['missing_ET'] < self.config['max_missing_mass'])
        
        return df[mask].copy(), mask
    
    def selection_efficiency(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute selection efficiency statistics."""
        N_total = len(mask)
        N_passed = np.sum(mask)
        
        return {
            'N_total': N_total,
            'N_passed': N_passed,
            'efficiency': N_passed / N_total if N_total > 0 else 0.0
        }


class TauPairClassifier:
    """
    Machine learning classifier for τ⁺τ⁻ entanglement.
    
    Three-class classification:
    - 0: Thermal (no entanglement)
    - 1: Mixed (partial entanglement)
    - 2: Entangled (quantum correlations dominant)
    
    Supports multiple ML backends:
    - RandomForest (fast, interpretable)
    - XGBoost (high accuracy, GPU support)
    - GradientBoosting (sklearn fallback)
    """
    
    def __init__(self, classifier_type: str = 'xgboost', n_estimators: int = 200):
        """
        Args:
            classifier_type: 'xgboost', 'random_forest', or 'gradient_boosting'
            n_estimators: Number of trees
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for classification")
        
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        
        # Initialize classifier
        self.model = self._create_model()
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Feature importance
        self.feature_importance = None
    
    def _create_model(self):
        """Create ML model based on type."""
        if self.classifier_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                tree_method='hist',
                random_state=42
            )
        elif self.classifier_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train classifier on labeled data.
        
        Args:
            X_train: Training features (N_events, N_features)
            y_train: Labels (0=thermal, 1=mixed, 2=entangled)
            X_val: Optional validation features
            y_val: Optional validation labels
        
        Returns:
            Training metrics dictionary
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            
            if self.classifier_type == 'xgboost':
                eval_set = [(X_val_scaled, y_val)]
                self.model.fit(X_train_scaled, y_train,
                             eval_set=eval_set,
                             verbose=False)
            else:
                self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        # Compute training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        train_accuracy = np.mean(y_pred_train == y_train)
        
        metrics = {'train_accuracy': train_accuracy}
        
        if X_val is not None:
            y_pred_val = self.model.predict(X_val_scaled)
            val_accuracy = np.mean(y_pred_val == y_val)
            metrics['val_accuracy'] = val_accuracy
            
            # Multi-class AUC
            try:
                y_proba_val = self.model.predict_proba(X_val_scaled)
                auc = roc_auc_score(y_val, y_proba_val, multi_class='ovr')
                metrics['val_auc'] = auc
            except:
                pass
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict entanglement class for new events.
        
        Args:
            X: Feature matrix (N_events, N_features)
        
        Returns:
            (predicted_classes, class_probabilities)
        """
        X_scaled = self.scaler.transform(X)
        
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        return y_pred, y_proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Returns:
            Dictionary with metrics and confusion matrix
        """
        y_pred, y_proba = self.predict(X_test)
        
        # Accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        report = classification_report(y_test, y_pred, 
                                      target_names=['Thermal', 'Mixed', 'Entangled'],
                                      output_dict=True)
        
        # Multi-class AUC
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except:
            auc = None
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'auc': auc
        }
    
    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        import joblib
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'classifier_type': self.classifier_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pre-trained model."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data.get('feature_importance')
        self.classifier_type = model_data.get('classifier_type', 'unknown')
        
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("=== Belle II Tau Pair Classifier Test ===\n")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    N_events = 10000
    
    # Simulated event data
    data = {
        'tau_plus_px': np.random.randn(N_events) * 2.0,
        'tau_plus_py': np.random.randn(N_events) * 2.0,
        'tau_plus_pz': np.random.randn(N_events) * 3.0,
        'tau_plus_E': np.random.uniform(3, 7, N_events),
        'tau_minus_px': np.random.randn(N_events) * 2.0,
        'tau_minus_py': np.random.randn(N_events) * 2.0,
        'tau_minus_pz': np.random.randn(N_events) * 3.0,
        'tau_minus_E': np.random.uniform(3, 7, N_events),
    }
    
    df = pd.DataFrame(data)
    
    # Extract features
    print("Extracting features...")
    extractor = EntanglementFeatureExtractor()
    features = extractor.extract_from_dataframe(df)
    
    print(f"Extracted {len(features.columns)} features")
    print(f"Feature names: {list(features.columns[:5])}...")
    
    # Apply kinematic selection
    print("\nApplying kinematic cuts...")
    selector = KinematicSelector()
    features_selected, mask = selector.apply_cuts(features)
    
    eff_stats = selector.selection_efficiency(mask)
    print(f"Selection efficiency: {eff_stats['efficiency']*100:.1f}%")
    print(f"Events passed: {eff_stats['N_passed']}/{eff_stats['N_total']}")
    
    # Generate synthetic labels for testing
    # Labels based on concurrence_estimate
    labels = np.zeros(len(features_selected), dtype=int)
    labels[features_selected['concurrence_estimate'] > 0.3] = 1  # Mixed
    labels[features_selected['concurrence_estimate'] > 0.6] = 2  # Entangled
    
    print(f"\nLabel distribution:")
    print(f"  Thermal: {np.sum(labels==0)}")
    print(f"  Mixed: {np.sum(labels==1)}")
    print(f"  Entangled: {np.sum(labels==2)}")
    
    if SKLEARN_AVAILABLE:
        # Train classifier
        print("\nTraining classifier...")
        X = features_selected.values
        y = labels
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        classifier = TauPairClassifier(classifier_type='random_forest', n_estimators=100)
        
        train_metrics = classifier.train(X_train, y_train, X_test, y_test)
        
        print(f"Train accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"Val accuracy: {train_metrics['val_accuracy']:.4f}")
        
        # Evaluate
        eval_results = classifier.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {eval_results['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(eval_results['confusion_matrix'])
