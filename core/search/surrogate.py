"""
Docstring for core.search.surrogate
bflow에서 low-prob 영역 skip 가능
"""

# core/search/surrogate.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_surrogate(df):
    model = RandomForestRegressor(n_estimators=50)
    X = df[["AWG","Parallels","Turns_per_slot_side","rpm"]]
    y = df["margin_pct"]
    model.fit(X,y)
    return model

def predict_margin(model, params):
    return model.predict([params])[0]

def perturb(point):
    awg, par, turns, rpm = point
    return (
        awg + np.random.randint(-1,2),
        par + np.random.randint(-2,3),
        turns + np.random.randint(-2,3),
        rpm
    )


"""
Advanced Surrogate Model for ESC Motor Design
Features: Gaussian Process Regression (GPR), Uncertainty-based Search (UCB),
          Multi-output prediction for Margin & Fill Factor.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

class DesignSurrogate:
    def __init__(self):
        # 커널 정의: 노이즈(WhiteKernel)와 비선형 패턴(RBF)을 결합
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10,
            alpha=0.1 # 수치적 안정성 확보
        )
        self.scaler_x = StandardScaler()
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """Pass 1 데이터를 학습하여 설계 공간의 지도를 생성"""
        # 입력 변수: AWG, 병렬수, 턴수, RPM
        X = df[["AWG", "Parallels", "Turns_per_slot_side", "rpm"]].values
        y = df["margin_pct"].values

        # 데이터 정규화 (GP 모델의 성능을 결정짓는 핵심 단계)
        X_scaled = self.scaler_x.fit_transform(X)
        
        print("[SURROGATE] Training Gaussian Process model...")
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(f"[SURROGATE] Training complete. Optimized Kernel: {self.model.kernel_}")

    def predict_with_uncertainty(self, params):
        """
        단순 예측이 아닌 '예측값'과 '신뢰도(표준편차)'를 함께 반환
        """
        X_new = np.array([params])
        X_scaled = self.scaler_x.transform(X_new)
        
        # return_std=True를 통해 AI의 불확실성을 측정
        mu, sigma = self.model.predict(X_scaled, return_std=True)
        return mu[0], sigma[0]

    def get_acquisition_value(self, params, kappa=2.576):
        """
        Upper Confidence Bound (UCB) 알고리즘:
        AI가 좋다고 판단하는 곳(mu) + AI가 잘 모르는 곳(sigma)을 동시에 고려
        """
        mu, sigma = self.predict_with_uncertainty(params)
        # kappa가 높을수록 '새로운 영역 탐색'에 비중을 둠
        return mu + kappa * sigma

def run_smart_narrowing(surrogate_model, candidates_df, top_n=500):
    """
    AI가 판단하기에 가장 유망한(UCB 기준) 후보들만 골라내는 가속기
    """
    if not surrogate_model.is_trained:
        return candidates_df

    scores = []
    for _, row in candidates_df.iterrows():
        params = [row["AWG"], row["Parallels"], row["Turns_per_slot_side"], row["rpm"]]
        score = surrogate_model.get_acquisition_value(params)
        scores.append(score)

    candidates_df["ai_score"] = scores
    # AI 점수 기준으로 정렬하여 상위 후보만 Pass 2로 전달
    refined_df = candidates_df.sort_values(by="ai_score", ascending=False).head(top_n)
    
    print(f"[NARROWING] AI narrowed candidates from {len(candidates_df)} to {len(refined_df)}")
    return refined_df

def accept(proposal, current, model):
    p_margin = predict_margin(model, proposal)
    c_margin = predict_margin(model, current)

    if p_margin > c_margin:
        return True

    prob = np.exp((p_margin - c_margin)/5.0)
    return np.random.rand() < prob

def mcmc_search(initial_point, model, steps=500):
    current = initial_point

    for _ in range(steps):
        proposal = perturb(current)
        if accept(proposal, current, model):
            current = proposal

    return current
