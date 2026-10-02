import numpy as np
import torch
import logging
from typing import Dict, Optional, Tuple, Any
from hdlnn.contracts.interfaces import IDivergenceScorer
from hdlnn.contracts.schemas import TrajectoryState, AnomalyDecision
from hdlnn.divergence.manifold import ReferenceManifold

logger = logging.getLogger(__name__)

class DivergenceScorer(IDivergenceScorer):
    """Computes behavioral drift anomalies using either Mahalanobis or Cosine EWMA modes."""
    
    def __init__(
        self,
        mode: str = "mahalanobis",
        threshold_k: float = 3.0,
        alpha: float = 0.05,
        hidden_dim: int = 64,
        warmup_steps: int = 0,
        model: Optional[Any] = None,
        n_clusters: int = 1,
        covariance_type: str = "full",
        trim_ratio: float = 0.02,
        use_likelihood_ratio: bool = False,
        lr_epsilon: float = 0.5
    ):
        self.mode = mode.lower()
        self.threshold_k = threshold_k
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.model = model
        self.initial_threshold = 1.0
        self.use_likelihood_ratio = use_likelihood_ratio
        self.lr_epsilon = lr_epsilon
        
        # Mahalanobis components
        self.manifold = ReferenceManifold(hidden_dim=hidden_dim, n_clusters=n_clusters, covariance_type=covariance_type, trim_ratio=trim_ratio)
        self.attack_manifold: Optional[ReferenceManifold] = None
        if self.use_likelihood_ratio:
            self.attack_manifold = ReferenceManifold(hidden_dim=hidden_dim, n_clusters=1, covariance_type="shared", trim_ratio=0.0)
        self.mahalanobis_threshold = 0.0
        
        # Discriminant subspace correction
        self.w_discriminant: Optional[torch.Tensor] = None
        self.z_shift: float = 0.0
        self.z_scale: float = 1.0
        
        # 66-D Calibrated Linear Decision Boundary: w_boundary in R^{66}, b_boundary in R
        self.w_boundary: Optional[torch.Tensor] = None
        self.b_boundary: float = 0.0
        
        # 5-D Calibrated Quadratic Boundary: [fish, lr, fish^2, lr^2, fish*lr]
        self.w_quad: Optional[torch.Tensor] = None
        self.b_quad: float = 0.0
        
        # Distilled Student MLP: 4D Fisher Subspace + 5 -> 4 -> 1 MLP
        self.w_sub_proj: Optional[torch.Tensor] = None # (64, 4)
        self.w1_student: Optional[torch.Tensor] = None # (4, 5)
        self.b1_student: Optional[torch.Tensor] = None # (4,)
        self.w2_student: Optional[torch.Tensor] = None # (1, 4)
        self.b2_student: float = 0.0
        
        # Production Teacher MLP: 66 -> 16 -> 1 direct execution
        self.w1_teacher: Optional[torch.Tensor] = None # (16, 66)
        self.b1_teacher: Optional[torch.Tensor] = None # (16,)
        self.w2_teacher: Optional[torch.Tensor] = None # (1, 16)
        self.b2_teacher: float = 0.0
        self.h_buf_teacher: Optional[torch.Tensor] = None # (16,) preallocated buffer
        
        # Boundary Discriminative Head: 66 -> 32 -> 16 -> 1
        self.w1_head: Optional[torch.Tensor] = None # (32, 66)
        self.b1_head: Optional[torch.Tensor] = None # (32,)
        self.w2_head: Optional[torch.Tensor] = None # (16, 32)
        self.b2_head: Optional[torch.Tensor] = None # (16,)
        self.w3_head: Optional[torch.Tensor] = None # (1, 16)
        self.b3_head: float = 0.0
        self.h1_buf_head: Optional[torch.Tensor] = None # (32,)
        self.h2_buf_head: Optional[torch.Tensor] = None # (16,)
        
        # Instantaneous-Temporal Fusion Head: [h_t(64), z_t(128), fish_h(1), fish_z(1), lr(1)] = 195 -> 16 -> 1
        self.w_discriminant_z: Optional[torch.Tensor] = None # (128,)
        self.w1_fusion: Optional[torch.Tensor] = None # (16, 195)
        self.b1_fusion: Optional[torch.Tensor] = None # (16,)
        self.w2_fusion: Optional[torch.Tensor] = None # (1, 16)
        self.b2_fusion: float = 0.0
        self.h_buf_fusion: Optional[torch.Tensor] = None # (16,)
        self.use_instantaneous_fusion: bool = False
        
        # Subspace Instantaneous-Temporal Fusion: [h_t(64), z'_t(16), fish_h(1), lr(1)] = 82 -> 16 -> 1
        self.w_sub_z: Optional[torch.Tensor] = None # (128, 16)
        self.is_subspace_fusion: bool = False
        
        # Teacher Residual & Gating Parameters
        self.gate_th: float = 1.0
        self.gate_margin: float = 0.0 # 0.0 means always run teacher if fitted; >0.0 gates by |lr - gate_th|/gate_th < margin
        self.residual_alpha: float = 0.0 # 0.0 means direct teacher or base; >0.0 blends: S_base + alpha * (S_teacher - S_base)
        self.base_lr_mean: float = 0.0
        self.base_lr_std: float = 1.0

        # Cosine EWMA tracking per entity: entity_id -> (mean, var, count)
        self.ewma_stats: Dict[str, Tuple[float, float, int]] = {}
        # Stores the current observed hypervector: entity_id -> torch.Tensor
        self.last_observed: Dict[str, torch.Tensor] = {}

    def fit_discriminant_subspace(self, normal_states: torch.Tensor, attack_states: torch.Tensor) -> None:
        """Computes the 1D Fisher linear discriminant direction between normal and attack states."""
        if len(normal_states) >= 10 and len(attack_states) >= 10:
            mu_n = normal_states.mean(0)
            mu_a = attack_states.mean(0)
            cov_n = torch.cov(normal_states.T)
            cov_a = torch.cov(attack_states.T)
            cov_pool = 0.5 * (cov_n + cov_a) + 1e-4 * torch.eye(self.hidden_dim, device=normal_states.device)
            try:
                w = torch.linalg.solve(cov_pool, (mu_a - mu_n).unsqueeze(1)).squeeze(1)
                w = w / torch.norm(w)
                self.w_discriminant = w
                
                # Calibration statistics on normal projections
                projs = normal_states @ w
                self.z_shift = float(projs.median().item())
                iqr = float((torch.quantile(projs, 0.75) - torch.quantile(projs, 0.25)).item())
                self.z_scale = iqr if iqr > 1e-4 else float(projs.std().item() + 1e-4)
                logger.info("Fitted 1D Fisher discriminant direction successfully.")
            except Exception as e:
                logger.warning("Failed to fit discriminant subspace: %s", e)

    def fit_quadratic_boundary(self, val_states: torch.Tensor, val_labels: np.ndarray) -> None:
        """Fits an exact 5-D quadratic decision boundary on [fish, lr, fish^2, lr^2, fish*lr]."""
        try:
            from sklearn.linear_model import LogisticRegression
            d_norm = self.manifold.compute_mahalanobis_distance(val_states)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states)
            lr = (d_norm / (d_att + self.lr_epsilon)).cpu().numpy()
            
            fish = (val_states @ self.w_discriminant).cpu().numpy() if self.w_discriminant is not None else np.zeros_like(lr)
            
            X_q = np.stack([fish, lr, fish**2, lr**2, fish * lr], axis=1)
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42).fit(X_q, val_labels)
            
            self.w_quad = torch.tensor(clf.coef_[0], dtype=torch.float32, device=val_states.device)
            self.b_quad = float(clf.intercept_[0])
            logger.info("Fitted 5-D quadratic decision boundary successfully.")
        except Exception as e:
            logger.warning("Failed to fit quadratic decision boundary: %s", e)

    def fit_distilled_student(self, val_states: torch.Tensor, val_labels: np.ndarray) -> None:
        """Fits an ultra-compact 4D-Subspace + 5 -> 4 -> 1 MLP student distilled from validation state geometry."""
        try:
            from sklearn.neural_network import MLPClassifier
            
            # Construct 4-D Fisher subspace
            norm_mask = (val_labels == 0)
            mu_n = val_states[norm_mask].mean(0).cpu().numpy()
            mu_a = val_states[~norm_mask].mean(0).cpu().numpy()
            cov_pool = 0.5 * (np.cov(val_states[norm_mask].cpu().numpy().T) + np.cov(val_states[~norm_mask].cpu().numpy().T)) + 1e-4 * np.eye(self.hidden_dim)
            w_fish1 = np.linalg.solve(cov_pool, mu_a - mu_n)
            w_fish1 = (w_fish1 / np.linalg.norm(w_fish1)).reshape(-1, 1)
            
            means_arr = np.stack([m.cpu().numpy() for m in self.manifold.means])
            diff_means = means_arr - mu_a
            U, S, Vt = np.linalg.svd(diff_means, full_matrices=False)
            W_sub = Vt[:3].T
            W_proj = np.hstack([w_fish1, W_sub])
            Q, _ = np.linalg.qr(W_proj) # (64, 4)
            self.w_sub_proj = torch.tensor(Q, dtype=torch.float32, device=val_states.device)
            
            # Form 5-D features: [proj_4D, lr_score]
            proj_val = val_states @ self.w_sub_proj
            d_norm = self.manifold.compute_mahalanobis_distance(val_states)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states)
            lr_scores = (d_norm / (d_att + self.lr_epsilon)).unsqueeze(1)
            X_5d = torch.cat([proj_val, lr_scores], dim=-1).cpu().numpy()
            
            # Train 5 -> 4 -> 1 MLP
            mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=42).fit(X_5d, val_labels)
            
            self.w1_student = torch.tensor(mlp.coefs_[0].T, dtype=torch.float32, device=val_states.device) # (4, 5)
            self.b1_student = torch.tensor(mlp.intercepts_[0], dtype=torch.float32, device=val_states.device) # (4,)
            self.w2_student = torch.tensor(mlp.coefs_[1].T, dtype=torch.float32, device=val_states.device) # (1, 4)
            self.b2_student = float(mlp.intercepts_[1][0])
            logger.info("Fitted 4D-Subspace + 5 -> 4 -> 1 Distilled Student MLP successfully.")
        except Exception as e:
            logger.warning("Failed to fit distilled student MLP: %s", e)

    def fit_quadratic_boundary(self, val_states: torch.Tensor, val_labels: np.ndarray) -> None:
        """Fits an exact 5-D quadratic decision boundary on [fish, lr, fish^2, lr^2, fish*lr]."""
        try:
            from sklearn.linear_model import LogisticRegression
            d_norm = self.manifold.compute_mahalanobis_distance(val_states)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states)
            lr = (d_norm / (d_att + self.lr_epsilon)).cpu().numpy()
            
            fish = (val_states @ self.w_discriminant).cpu().numpy() if self.w_discriminant is not None else np.zeros_like(lr)
            
            X_q = np.stack([fish, lr, fish**2, lr**2, fish * lr], axis=1)
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42).fit(X_q, val_labels)
            
            self.w_quad = torch.tensor(clf.coef_[0], dtype=torch.float32, device=val_states.device)
            self.b_quad = float(clf.intercept_[0])
            logger.info("Fitted 5-D quadratic decision boundary successfully.")
        except Exception as e:
            logger.warning("Failed to fit quadratic decision boundary: %s", e)

    def fit_linear_boundary(self, val_states: torch.Tensor, val_labels: np.ndarray) -> None:
        """Fits an exact 66-D linear decision boundary on [h_t, fish_proj, lr_score] using Logistic Regression."""
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Compute features for all validation states
            d_norm = self.manifold.compute_mahalanobis_distance(val_states)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states)
            lr_scores = (d_norm / (d_att + self.lr_epsilon)).unsqueeze(1)
            
            fish_proj = (val_states @ self.w_discriminant).unsqueeze(1) if self.w_discriminant is not None else torch.zeros_like(lr_scores)
            
            X_val = torch.cat([val_states, fish_proj, lr_scores], dim=-1).cpu().numpy()
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42).fit(X_val, val_labels)
            
            self.w_boundary = torch.tensor(clf.coef_[0], dtype=torch.float32, device=val_states.device)
            self.b_boundary = float(clf.intercept_[0])
            logger.info("Fitted 66-D calibrated linear decision boundary successfully.")
        except Exception as e:
            logger.warning("Failed to fit linear decision boundary: %s", e)

    def fit_teacher_mlp(self, val_states: torch.Tensor, val_labels: np.ndarray) -> None:
        """Fits the exact 66 -> 16 -> 1 Teacher MLP on [h_t, fish_proj, lr_score]."""
        try:
            from sklearn.neural_network import MLPClassifier
            d_norm = self.manifold.compute_mahalanobis_distance(val_states)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states)
            lr_scores = (d_norm / (d_att + self.lr_epsilon)).unsqueeze(1)
            fish_proj = (val_states @ self.w_discriminant).unsqueeze(1) if self.w_discriminant is not None else torch.zeros_like(lr_scores)
            X_val = torch.cat([val_states, fish_proj, lr_scores], dim=-1).cpu().numpy()
            
            mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42).fit(X_val, val_labels)
            
            self.w1_teacher = torch.tensor(mlp.coefs_[0].T, dtype=torch.float32, device=val_states.device) # (16, 66)
            self.b1_teacher = torch.tensor(mlp.intercepts_[0], dtype=torch.float32, device=val_states.device) # (16,)
            self.w2_teacher = torch.tensor(mlp.coefs_[1].T, dtype=torch.float32, device=val_states.device) # (1, 16)
            self.b2_teacher = float(mlp.intercepts_[1][0])
            self.h_buf_teacher = torch.empty(16, dtype=torch.float32, device=val_states.device)
            logger.info("Fitted Production 66 -> 16 -> 1 Teacher MLP successfully.")
        except Exception as e:
            logger.warning("Failed to fit Teacher MLP: %s", e)

    def fit_discriminative_head(self, val_states: torch.Tensor, val_labels: np.ndarray) -> None:
        """Fits the boundary-focused 66 -> 32 -> 16 -> 1 Discriminative Head on [h_t, fish_proj, lr_score]."""
        try:
            from sklearn.neural_network import MLPClassifier
            d_norm = self.manifold.compute_mahalanobis_distance(val_states)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states)
            lr_scores = (d_norm / (d_att + self.lr_epsilon)).unsqueeze(1)
            fish_proj = (val_states @ self.w_discriminant).unsqueeze(1) if self.w_discriminant is not None else torch.zeros_like(lr_scores)
            X_val = torch.cat([val_states, fish_proj, lr_scores], dim=-1).cpu().numpy()
            
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42).fit(X_val, val_labels)
            
            self.w1_head = torch.tensor(mlp.coefs_[0].T, dtype=torch.float32, device=val_states.device) # (32, 66)
            self.b1_head = torch.tensor(mlp.intercepts_[0], dtype=torch.float32, device=val_states.device) # (32,)
            self.w2_head = torch.tensor(mlp.coefs_[1].T, dtype=torch.float32, device=val_states.device) # (16, 32)
            self.b2_head = torch.tensor(mlp.intercepts_[1], dtype=torch.float32, device=val_states.device) # (16,)
            self.w3_head = torch.tensor(mlp.coefs_[2].T, dtype=torch.float32, device=val_states.device) # (1, 16)
            self.b3_head = float(mlp.intercepts_[2][0])
            self.h1_buf_head = torch.empty(32, dtype=torch.float32, device=val_states.device)
            self.h2_buf_head = torch.empty(16, dtype=torch.float32, device=val_states.device)
            logger.info("Fitted 66 -> 32 -> 16 -> 1 Discriminative Head successfully.")
        except Exception as e:
            logger.warning("Failed to fit Discriminative Head: %s", e)

    def fit_fusion_head(
        self, 
        val_states_h: torch.Tensor, 
        val_states_z: torch.Tensor, 
        val_labels: np.ndarray,
        subspace_dim: Optional[int] = None
    ) -> None:
        """Fits the Instantaneous-Temporal Fusion Head on [h_t, z_t, fish_h, fish_z, lr].
        
        Args:
            val_states_h: (N, 64) temporal liquid states
            val_states_z: (N, 128) instantaneous backbone projections
            val_labels: (N,) binary labels (0=normal, 1=attack)
            subspace_dim: Optional subspace dimension for z (e.g. 16)
        """
        try:
            from sklearn.neural_network import MLPClassifier
            
            # 1. Likelihood Ratio on liquid state
            d_norm = self.manifold.compute_mahalanobis_distance(val_states_h)
            d_att = self.attack_manifold.compute_mahalanobis_distance(val_states_h)
            lr_scores = (d_norm / (d_att + self.lr_epsilon)).unsqueeze(1) # (N, 1)
            
            # 2. Fisher projection on liquid state h
            fish_h = (val_states_h @ self.w_discriminant).unsqueeze(1) if self.w_discriminant is not None else torch.zeros_like(lr_scores)
            
            # 3. Fisher projection on instantaneous projection z
            norm_mask = (val_labels == 0)
            att_mask = (val_labels == 1)
            if np.sum(norm_mask) >= 10 and np.sum(att_mask) >= 10:
                z_norm = val_states_z[norm_mask]
                z_att = val_states_z[att_mask]
                mu_zn = z_norm.mean(0)
                mu_za = z_att.mean(0)
                cov_zn = torch.cov(z_norm.T)
                cov_za = torch.cov(z_att.T)
                cov_z_pool = 0.5 * (cov_zn + cov_za) + 1e-4 * torch.eye(val_states_z.size(-1), device=val_states_z.device)
                w_z = torch.linalg.solve(cov_z_pool, (mu_za - mu_zn).unsqueeze(1)).squeeze(1)
                w_z = w_z / torch.norm(w_z)
                self.w_discriminant_z = w_z
                fish_z = (val_states_z @ self.w_discriminant_z).unsqueeze(1)
            else:
                fish_z = torch.zeros_like(lr_scores)
                
            if subspace_dim is not None and subspace_dim > 0:
                # Subspace reduction on z
                from sklearn.decomposition import PCA
                pca = PCA(n_components=subspace_dim, random_state=42)
                z_proj_np = pca.fit_transform(val_states_z.cpu().numpy())
                self.w_sub_z = torch.tensor(pca.components_.T, dtype=torch.float32, device=val_states_h.device) # (128, subspace_dim)
                self.is_subspace_fusion = True
                z_feature = torch.tensor(z_proj_np, dtype=torch.float32, device=val_states_h.device)
                X_feat = torch.cat([val_states_h, z_feature, fish_h, fish_z, lr_scores], dim=-1).cpu().numpy()
            elif subspace_dim == 0:
                # 67-D Compact Canonical Fusion: [h_t(64), fish_h(1), lr(1), fish_z(1)]
                self.is_subspace_fusion = False
                self.w_sub_z = None
                X_feat = torch.cat([val_states_h, fish_h, lr_scores, fish_z], dim=-1).cpu().numpy()
            else:
                # Full 195-D Fusion: [h_t(64), z_t(128), fish_h(1), fish_z(1), lr(1)]
                self.is_subspace_fusion = False
                self.w_sub_z = None
                X_feat = torch.cat([val_states_h, val_states_z, fish_h, fish_z, lr_scores], dim=-1).cpu().numpy()
                
            if subspace_dim == -1:
                # Direct 192-D Linear Boundary: [h_t(64), z_t(128)]
                from sklearn.linear_model import LogisticRegression
                X_feat = torch.cat([val_states_h, val_states_z], dim=-1).cpu().numpy()
                in_dim = X_feat.shape[1]
                clf = LogisticRegression(max_iter=500, C=1.0, random_state=42).fit(X_feat, val_labels)
                self.w1_fusion = torch.tensor(clf.coef_, dtype=torch.float32, device=val_states_h.device) # (1, 192)
                self.b1_fusion = torch.tensor(clf.intercept_, dtype=torch.float32, device=val_states_h.device) # (1,)
                self.w2_fusion = None
                self.b2_fusion = 0.0
                self.is_subspace_fusion = False
                self.w_sub_z = None
                self.use_instantaneous_fusion = True
                logger.info("Fitted Linear Instantaneous-Temporal Fusion Boundary (%d -> 1) successfully.", in_dim)
            else:
                in_dim = X_feat.shape[1]
                mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42).fit(X_feat, val_labels)
                
                self.w1_fusion = torch.tensor(mlp.coefs_[0].T, dtype=torch.float32, device=val_states_h.device) # (16, in_dim)
                self.b1_fusion = torch.tensor(mlp.intercepts_[0], dtype=torch.float32, device=val_states_h.device) # (16,)
                self.w2_fusion = torch.tensor(mlp.coefs_[1].T, dtype=torch.float32, device=val_states_h.device) # (1, 16)
                self.b2_fusion = float(mlp.intercepts_[1][0])
                self.h_buf_fusion = torch.empty(16, dtype=torch.float32, device=val_states_h.device)
                self.use_instantaneous_fusion = True
                logger.info("Fitted Instantaneous-Temporal Fusion Head (%d -> 16 -> 1) successfully.", in_dim)
        except Exception as e:
            logger.warning("Failed to fit Instantaneous-Temporal Fusion Head: %s", e)

    def compute_fusion_score(self, h_t: torch.Tensor, z_t: torch.Tensor) -> float:
        """Evaluates the Instantaneous-Temporal Fusion score in-place."""
        if not self.use_instantaneous_fusion or self.w1_fusion is None:
            return self.compute_score(h_t)
            
        d_norm = self.manifold.compute_mahalanobis_distance(h_t)
        d_att = self.attack_manifold.compute_mahalanobis_distance(h_t)
        lr_score = d_norm / (d_att + self.lr_epsilon)
        
        vec_h = h_t.view(-1)
        vec_z = z_t.view(-1)
        
        fish_h = torch.dot(vec_h, self.w_discriminant).view(1) if self.w_discriminant is not None else torch.zeros(1, device=h_t.device)
        fish_z = torch.dot(vec_z, self.w_discriminant_z).view(1) if self.w_discriminant_z is not None else torch.zeros(1, device=h_t.device)
        
        if self.w1_fusion.size(1) == 192:
            feat = torch.cat([vec_h, vec_z], dim=0)
        elif self.w1_fusion.size(1) == 67:
            feat = torch.cat([vec_h, fish_h, lr_score.view(1), fish_z], dim=0)
        elif self.is_subspace_fusion and self.w_sub_z is not None:
            z_proj = torch.mv(self.w_sub_z.t(), vec_z)
            feat = torch.cat([vec_h, z_proj, fish_h, fish_z, lr_score.view(1)], dim=0)
        else:
            feat = torch.cat([vec_h, vec_z, fish_h, fish_z, lr_score.view(1)], dim=0)
            
        if self.w2_fusion is None:
            # Linear boundary
            logit = torch.dot(self.w1_fusion[0], feat) + self.b1_fusion[0]
            prob_f = float(torch.sigmoid(logit).item())
        else:
            if self.h_buf_fusion is None:
                self.h_buf_fusion = torch.empty(16, dtype=torch.float32, device=h_t.device)
            torch.mv(self.w1_fusion, feat, out=self.h_buf_fusion)
            self.h_buf_fusion.add_(self.b1_fusion)
            torch.clamp_min_(self.h_buf_fusion, 0.0)
            logit = torch.dot(self.h_buf_fusion, self.w2_fusion[0]) + self.b2_fusion
            prob_f = float(torch.sigmoid(logit).item())
            
        if self.residual_alpha > 0.0:
            lr_val = float(lr_score.item())
            base_prob = 1.0 / (1.0 + np.exp(-(lr_val - self.base_lr_mean) / (self.base_lr_std + 1e-5)))
            return float(base_prob + self.residual_alpha * (prob_f - base_prob))
        return prob_f

    def fit_attack_manifold(self, attack_states: torch.Tensor) -> None:
        """Fits an attack reference manifold using validation anomaly states."""
        if self.attack_manifold is not None and len(attack_states) >= 10:
            self.attack_manifold.fit(attack_states)
            logger.info("Fitted attack reference manifold on %d anomaly states.", len(attack_states))

    def compute_score(self, state_to_score: torch.Tensor) -> float:
        """Computes the anomaly score (Mahalanobis distance, Likelihood-Ratio, 66-D Boundary, 5-D Quadratic, Distilled Student, Teacher MLP, or Discriminative Head)."""
        d_norm = self.manifold.compute_mahalanobis_distance(state_to_score)
        if self.use_likelihood_ratio and self.attack_manifold is not None and self.attack_manifold.is_fit:
            d_att = self.attack_manifold.compute_mahalanobis_distance(state_to_score)
            lr_score = d_norm / (d_att + self.lr_epsilon)
            
            # If Boundary Discriminative Head is fitted, compute exact in-place score
            if self.w1_head is not None and self.w_discriminant is not None:
                vec = state_to_score.view(-1)
                fish_p = torch.dot(vec, self.w_discriminant)
                feat66 = torch.cat([vec, fish_p.view(1), lr_score.view(1)], dim=0)
                if self.h1_buf_head is None:
                    self.h1_buf_head = torch.empty(32, dtype=torch.float32, device=state_to_score.device)
                    self.h2_buf_head = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                torch.mv(self.w1_head, feat66, out=self.h1_buf_head)
                self.h1_buf_head.add_(self.b1_head)
                torch.clamp_min_(self.h1_buf_head, 0.0)
                
                torch.mv(self.w2_head, self.h1_buf_head, out=self.h2_buf_head)
                self.h2_buf_head.add_(self.b2_head)
                torch.clamp_min_(self.h2_buf_head, 0.0)
                
                logit = torch.dot(self.h2_buf_head, self.w3_head[0]) + self.b3_head
                return float(torch.sigmoid(logit).item())
                
            # If Production Teacher MLP is fitted, compute exact in-place nonlinear score
            if self.w1_teacher is not None and self.w_discriminant is not None:
                lr_val = float(lr_score.item())
                # If teacher gating is active and flow is outside ambiguous margin, return base LR score
                if self.gate_margin > 0.0:
                    rel_diff = abs(lr_val - self.gate_th) / max(self.gate_th, 1e-4)
                    if rel_diff >= self.gate_margin:
                        return lr_val
                        
                vec = state_to_score.view(-1)
                fish_p = torch.dot(vec, self.w_discriminant)
                feat66 = torch.cat([vec, fish_p.view(1), lr_score.view(1)], dim=0)
                if self.h_buf_teacher is None:
                    self.h_buf_teacher = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                torch.mv(self.w1_teacher, feat66, out=self.h_buf_teacher)
                self.h_buf_teacher.add_(self.b1_teacher)
                torch.clamp_min_(self.h_buf_teacher, 0.0)
                logit = torch.dot(self.h_buf_teacher, self.w2_teacher[0]) + self.b2_teacher
                prob_t = float(torch.sigmoid(logit).item())
                
                if self.residual_alpha > 0.0:
                    base_prob = 1.0 / (1.0 + np.exp(-(lr_val - self.base_lr_mean) / (self.base_lr_std + 1e-5)))
                    return float(base_prob + self.residual_alpha * (prob_t - base_prob))
                return prob_t
                
            # If Distilled Student MLP is fitted, compute exact nonlinear score
            if self.w_sub_proj is not None and self.w1_student is not None:
                vec = state_to_score.view(-1)
                proj4 = torch.mv(self.w_sub_proj.t(), vec)
                feat5 = torch.cat([proj4, lr_score.view(1)])
                h = torch.relu(torch.mv(self.w1_student, feat5) + self.b1_student)
                logit = torch.dot(h, self.w2_student[0]) + self.b2_student
                return float(torch.sigmoid(logit).item())
                
            # If 5-D Quadratic Boundary is fitted, compute exact distilled nonlinear score
            if self.w_quad is not None and self.w_discriminant is not None:
                vec = state_to_score.view(-1)
                fish = torch.dot(vec, self.w_discriminant)
                lr_val = lr_score.squeeze()
                feat5 = torch.stack([fish, lr_val, fish * fish, lr_val * lr_val, fish * lr_val])
                logit = torch.dot(feat5, self.w_quad) + self.b_quad
                return float(torch.sigmoid(logit).item())
                
            # If calibrated 66-D boundary is fitted, compute exact probability
            if self.w_boundary is not None and self.w_discriminant is not None:
                vec = state_to_score.view(-1)
                fish_p = torch.dot(vec, self.w_discriminant).unsqueeze(0)
                feat = torch.cat([vec, fish_p, lr_score.view(1)], dim=0)
                logit = torch.dot(feat, self.w_boundary) + self.b_boundary
                prob = torch.sigmoid(logit)
                return float(prob.item())
                
            # Fallback to Discriminant Likelihood-Ratio boost
            score = lr_score
            if self.w_discriminant is not None:
                vec = state_to_score.view(-1)
                proj = torch.dot(vec, self.w_discriminant)
                z = (proj - self.z_shift) / self.z_scale
                sig = torch.sigmoid(z)
                score = score * (1.0 + 0.5 * sig)
                
            return float(score.item())
        return float(d_norm.item())

    def register_observed(self, entity_id: str, vector: torch.Tensor) -> None:
        """Registers the actual observed hypervector for a given entity, required for Cosine mode."""
        self.last_observed[entity_id] = vector

    def fit_mahalanobis_threshold(self, normal_states: torch.Tensor) -> None:
        """Fits the ReferenceManifold and computes the static Mahalanobis threshold on normal validation data."""
        if self.mode != "mahalanobis":
            return
            
        # Fit the manifold
        self.manifold.fit(normal_states)
        
        # Calculate distances of the fitting normal states to set the threshold
        distances = self.manifold.compute_mahalanobis_distance(normal_states)
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        
        # Static threshold: mean + k * std
        self.mahalanobis_threshold = mean_dist + self.threshold_k * std_dist
        logger.info(
            f"Mahalanobis threshold fit: mean={mean_dist:.4f}, std={std_dist:.4f}, "
            f"k={self.threshold_k}, final_threshold={self.mahalanobis_threshold:.4f}"
        )

    def fit_f1_max_threshold(self, normal_states: torch.Tensor, val_states: torch.Tensor, val_labels: np.ndarray) -> float:
        """Fits ReferenceManifold on normal states and calibrates F1-maximizing threshold on validation split."""
        if self.mode != "mahalanobis":
            return self.mahalanobis_threshold
            
        # Fit the manifold
        self.manifold.fit(normal_states)
        
        # Score validation states
        val_dists = self.manifold.compute_mahalanobis_distance(val_states).detach().cpu().numpy()

        from hdlnn.eval.calibration import calibrate_threshold
        best_th, summary = calibrate_threshold(val_labels, val_dists, method="f1_max")
        self.mahalanobis_threshold = best_th
        logger.info("Validation F1-Max Mahalanobis threshold calibrated: threshold=%.4f, val_f1=%.4f",
                    best_th, summary["selected"]["f1"])
        return best_th

    def score(self, state: TrajectoryState) -> AnomalyDecision:
        """Computes behavioral drift against reference manifold or LNN next-state predictions."""
        if self.mode == "mahalanobis":
            return self._score_mahalanobis(state)
        elif self.mode == "cosine":
            return self._score_cosine_ewma(state)
        else:
            raise ValueError(f"Unsupported scoring mode: {self.mode}")

    def _score_mahalanobis(self, state: TrajectoryState) -> AnomalyDecision:
        # Calculate Mahalanobis distance
        # state.state is of shape [hidden_dim]
        dist_tensor = self.manifold.compute_mahalanobis_distance(state.state)
        drift_score = float(dist_tensor.item())
        
        is_anomaly = drift_score >= self.mahalanobis_threshold
        
        return AnomalyDecision(
            drift_score=drift_score,
            threshold=self.mahalanobis_threshold,
            is_anomaly=is_anomaly,
            scoring_mode="mahalanobis"
        )

    def _score_cosine_ewma(self, state: TrajectoryState) -> AnomalyDecision:
        if self.model is None:
            raise ValueError("LNN Sequence Model must be supplied to DivergenceScorer for Cosine mode.")
            
        entity_id = state.entity_id
        if entity_id not in self.last_observed:
            # Fallback if no observed hypervector is registered yet
            return AnomalyDecision(
                drift_score=0.0,
                threshold=1.0,
                is_anomaly=False,
                scoring_mode="cosine"
            )
            
        observed_hv = self.last_observed[entity_id]
        
        # 1. Project LNN state back to hypervector space
        self.model.eval()
        with torch.no_grad():
            predicted_hv = self.model.predict_next_vector(state.state.unsqueeze(0)).squeeze(0)
            
        # 2. Compute Cosine Distance: 1 - CosineSimilarity
        similarity = torch.cosine_similarity(observed_hv.unsqueeze(0), predicted_hv.unsqueeze(0))
        cosine_dist = float(1.0 - similarity.item())

        # 3. Update adaptive EWMA threshold
        stats = self.ewma_stats.get(entity_id, (0.0, 0.0, 0))
        mean_ewma, var_ewma, count = stats
        count += 1
        
        if count == 1:
            # Initialize statistics
            mean_ewma = cosine_dist
            var_ewma = 0.0
            threshold = self.initial_threshold
            is_anomaly = (cosine_dist > threshold) if (self.warmup_steps == 0 and threshold < 1.0) else False
        else:
            diff = cosine_dist - mean_ewma
            # Update mean EWMA
            mean_ewma = mean_ewma + self.alpha * diff
            # Update variance EWMA
            var_ewma = (1 - self.alpha) * (var_ewma + self.alpha * (diff ** 2))
            
            std_ewma = (var_ewma ** 0.5)
            threshold = mean_ewma + self.threshold_k * std_ewma
            
            # Anomaly verdict (suppressed during warmup period)
            if count <= self.warmup_steps:
                is_anomaly = False
            else:
                is_anomaly = cosine_dist > threshold

        # Save stats
        self.ewma_stats[entity_id] = (mean_ewma, var_ewma, count)
        
        return AnomalyDecision(
            drift_score=cosine_dist,
            threshold=threshold,
            is_anomaly=is_anomaly,
            scoring_mode="cosine"
        )
