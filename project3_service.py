# -*- coding: utf-8 -*-
"""
Project3 : PECVD 가상 계측(VM) · 이상 감지(FDC) - 서비스(계산) 모듈
  * 강사님 project3_service.py 구조(함수 이름·후보 모델·하이퍼파라미터)를 기반으로
  * [공정] 표시 = 공정 조건을 반영해 새로 추가/변경한 부분

  공정 조건 반영 포인트
  1. 계측 시점 기준 변수 선별 : 증착 '전/중'에 알 수 있는 값만 입력 (POST·EDGE CENTER·식별자 제외)
  2. 표준 레시피(Setpoint) 대비 이탈량 · 이탈 개수 변수 추가
     - 공정 시간은 가스별 표준 시간(SiH4 120s / TEOS 60s) 기준으로 이탈 판단
  3. RF 반사율(%) = 반사 전력 / 인가 전력 (플라즈마 매칭 상태) , CMP 연마량 추가
  4. 품질 지표 : 증착 속도(Å/s), 두께 균일도(%) → EDA 용 (계측 후 값이라 입력에는 사용 안 함)
  5. 불량 기준 : 두께는 너무 얇아도/두꺼워도 불량(양측 3σ), 파티클은 많을 때만 불량(상측 3σ)
  6. 실시간 FDC : 투입 전 레시피 인터락 + SPC 룰(3σ 이탈 / 2σ 연속 / 7점 한쪽 쏠림)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
warnings.filterwarnings('ignore')

#  pip install scikit-learn xgboost lightgbm seaborn streamlit

# ---------------- 한글 폰트 (Streamlit Cloud = 나눔고딕 / Windows = 맑은고딕) ----------------
nanum_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(nanum_path):
    fm.fontManager.addfont(nanum_path)
    plt.rc('font', family='NanumGothic')
else:
    try:
        plt.rc('font', family='Malgun Gothic')
    except Exception:
        plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================================
# [공정] 공정 조건 정의
# =====================================================================
# 표준 레시피 설정값 (데이터의 최빈값 = 실제 운영 레시피)
RECIPE_SETPOINT = {
    'RF Power': 300,                      # W
    'Chuck center temp': 120,             # ℃
    'Chamber pressure': 1200,             # mTorr
    'Precursor flow rate': 50,            # sccm
    'ashing time': 300,                   # s
    'pre clean chemical flow rate': 1000, # 세정 약품 유량
}
RECIPE_TOL = 0.02                          # 설정값 ±2% 넘으면 '레시피 이탈'
PROCESS_TIME_STD = {'SiH4': 120, 'TEOS': 60}   # 가스별 표준 증착 시간(s)

# 증착 전/중에 알 수 있는 공정 변수 (모델 입력 후보)
PROCESS_COLS = ['SiO2 Under SiN depo THK', 'SiO2 Under SiN CMP THK',          # 이전 공정
                'etch RF reflect power', 'ashing time', 'pre clean chemical flow rate',  # 전처리
                'PRE PC', 'PRE THK AVG',                                          # 증착 전 계측
                'Chuck center temp', 'Chamber pressure', 'Precursor flow rate',   # 메인 증착
                'RF Power', 'RF reflect Power Max', 'Main process time']
ENGINEERED_COLS = [c + '_dev' for c in RECIPE_SETPOINT] + ['time_dev', 'Recipe_Dev_Count',
                                                          'RF_Reflect_Ratio', 'CMP_Removal']
CAT_COLS = ['Chamber', 'Precursor']
# 계측 후 값 / 식별자 → 입력 금지 (데이터 누수)
LEAK_COLS = ['POST THK AVG', 'POST PC', 'EDGE CENTER', 'LOT', 'WF', 'TKIN time']

TARGETS = ['Depo_THK', 'Particle']


def find_csv(name='semi_process_data2.csv'):
    """data 폴더 / 저장소 첫 화면 / 파일명 달라도 자동으로 csv 찾기"""
    for p in [os.path.join(BASE_DIR, 'data', name), os.path.join(BASE_DIR, name)]:
        if os.path.exists(p):
            return p
    found = glob.glob(os.path.join(BASE_DIR, '**', '*.csv'), recursive=True)
    if not found:
        raise FileNotFoundError('csv 없음. 폴더 목록: ' + str(os.listdir(BASE_DIR)))
    return found[0]


# %% 1. 데이터 로드 + 파생변수 + 라벨링
def load_csv(CSV_PATH=None):
    df = pd.read_csv(CSV_PATH or find_csv(), encoding='utf-8')

    # 1. 결측치 / 단일값 컬럼 삭제 (Chuck Edge temp = 100 고정)
    df = df.dropna(axis=1)
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() <= 1])

    # 2. Target : 이번 공정에서 쌓인 두께 / 새로 생긴 파티클
    df['Depo_THK'] = df['POST THK AVG'] - df['PRE THK AVG']
    df['Particle'] = (df['POST PC'] - df['PRE PC']).clip(lower=0)

    # 3. [공정] 레시피 이탈 변수
    for c, sp in RECIPE_SETPOINT.items():
        df[c + '_dev'] = df[c] - sp
    df['time_dev'] = df['Main process time'] - df['Precursor'].map(PROCESS_TIME_STD)
    off = [(df[c + '_dev'].abs() > sp * RECIPE_TOL) for c, sp in RECIPE_SETPOINT.items()]
    off.append(df['time_dev'] != 0)
    df['Recipe_Dev_Count'] = np.sum(off, axis=0)

    # 4. [공정] 플라즈마 매칭 · 연마량
    df['RF_Reflect_Ratio'] = df['RF reflect Power Max'] / df['RF Power'] * 100   # %
    df['CMP_Removal'] = df['SiO2 Under SiN depo THK'] - df['SiO2 Under SiN CMP THK']

    # 5. [공정] 품질 지표 (계측 후 값 → EDA 전용, 모델 입력 X)
    df['Depo_Rate'] = df['Depo_THK'] / df['Main process time']          # Å/s
    df['Uniformity'] = df['EDGE CENTER'] / df['Depo_THK'] * 100         # %, 작을수록 균일

    # 6. [공정] 불량 라벨 : 두께 = 양측 3σ, 파티클 = 상측 3σ (먼지는 적을수록 좋음)
    m, s = df['Depo_THK'].mean(), df['Depo_THK'].std()
    df['Depo_THK_y'] = ((df['Depo_THK'] < m - 3 * s) | (df['Depo_THK'] > m + 3 * s)).astype(int)
    m, s = df['Particle'].mean(), df['Particle'].std()
    df['Particle_y'] = (df['Particle'] > m + 3 * s).astype(int)
    return df


def spec_limits(df, target):
    m, s = df[target].mean(), df[target].std()
    lcl = m - 3 * s if target == 'Depo_THK' else None     # 파티클은 하한 없음
    return m, s, m + 3 * s, lcl


def make_features(df, feature_cols=None):
    """[공정] 증착 전/중 변수 + 레시피 이탈 변수 + 챔버·가스 원-핫"""
    X = df[PROCESS_COLS + ENGINEERED_COLS + CAT_COLS].copy()
    X = pd.get_dummies(X, columns=CAT_COLS, dtype=int)
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0)
    return X


def split_data(X, y, groups, mode, stratify=None):
    """mode='random' : 강사님 방식(무작위 8:2) / 'lot' : [공정] LOT 단위 분할 (처음 보는 LOT 로 평가)"""
    if mode == 'lot':
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)


# %% 2. EDA (강사님 코드 기반)
def _grid(n, wid=3):
    heig = (n + wid - 1) // wid
    fig, axes = plt.subplots(heig, wid, figsize=(12, 2.6 * heig))
    return fig, np.array(axes).reshape(-1)


def hist_pro(df_number):
    fig, axes = _grid(len(df_number.columns))
    for ax, col in zip(axes, df_number.columns):
        t_mean, t_std = df_number[col].mean(), df_number[col].std()
        ucl, lcl = t_mean + 3 * t_std, t_mean - 3 * t_std
        out = (df_number[col].max() > ucl) or (df_number[col].min() < lcl)
        ax.set_title(col + (': 이상치 확인' if out else ''), fontsize=8, color='red' if out else 'black')
        ax.hist(df_number[col], bins=30, color='skyblue', edgecolor='black')
        ax.axvline(t_mean, color='red', linestyle='--', linewidth=1)
        ax.axvline(ucl, color='orange', linestyle=':', linewidth=1)
        ax.axvline(lcl, color='orange', linestyle=':', linewidth=1)
    for ax in axes[len(df_number.columns):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def line_pro(df_number):
    fig, axes = _grid(len(df_number.columns))
    for ax, col in zip(axes, df_number.columns):
        t_mean, t_std = df_number[col].mean(), df_number[col].std()
        ucl, lcl = t_mean + 3 * t_std, t_mean - 3 * t_std
        out = (df_number[col].max() > ucl) or (df_number[col].min() < lcl)
        ax.set_title(col + (': 이상치 확인' if out else ''), fontsize=8, color='red' if out else 'black')
        ax.plot(range(len(df_number)), df_number[col], linewidth=0.5)
        ax.axhline(ucl, color='r', linestyle='dashed')
        ax.axhline(lcl, color='r', linestyle='dashed')
    for ax in axes[len(df_number.columns):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def corr_pro(df, target):
    cols = PROCESS_COLS + ['Recipe_Dev_Count', 'RF_Reflect_Ratio', 'CMP_Removal', target]
    corr = df[cols].corr()[target].abs().sort_values(ascending=False)
    sorted_cols = corr.index
    fig, ax = plt.subplots(2, 1, figsize=(11, 16))
    sns.heatmap(df[sorted_cols].corr(), annot=True, fmt='.2f', cmap='YlGnBu', ax=ax[0],
                linewidths=0.5, cbar_kws={'shrink': .8}, annot_kws={'size': 7})
    ax[0].set_title(f'공정 변수 간 상관관계 (정렬 기준: {target})', fontsize=14)
    ax[1].barh(sorted_cols[1:][::-1], corr.values[1:][::-1], color='#3182bd')
    ax[1].set_title(f'{target} 와의 상관계수 |r|')
    fig.tight_layout()
    return fig, sorted_cols[1:]


# %% 3. [공정] 공정 조건 분석
def recipe_check(df):
    """레시피 파라미터별 이탈 웨이퍼 수와 불량률"""
    rows = []
    for c, sp in RECIPE_SETPOINT.items():
        off = df[c + '_dev'].abs() > sp * RECIPE_TOL
        rows.append((c, f'{sp}', off))
    rows.append(('Main process time', 'SiH4 120 / TEOS 60', df['time_dev'] != 0))
    out = []
    for name, sp, off in rows:
        out.append({'파라미터': name, '표준값': sp, '이탈 웨이퍼': int(off.sum()),
                    '이탈 시 두께 불량률(%)': round(df.loc[off, 'Depo_THK_y'].mean() * 100, 1) if off.any() else 0,
                    '이탈 시 파티클 불량률(%)': round(df.loc[off, 'Particle_y'].mean() * 100, 1) if off.any() else 0})
    ok = df['Recipe_Dev_Count'] == 0
    out.append({'파라미터': '(참고) 레시피 정상', '표준값': '-', '이탈 웨이퍼': int(ok.sum()),
                '이탈 시 두께 불량률(%)': round(df.loc[ok, 'Depo_THK_y'].mean() * 100, 1),
                '이탈 시 파티클 불량률(%)': round(df.loc[ok, 'Particle_y'].mean() * 100, 1)})
    return pd.DataFrame(out)


def process_pro(df):
    """챔버·가스별 두께/파티클/증착속도/균일도 + 레시피 이탈 개수별 불량률"""
    fig, ax = plt.subplots(3, 2, figsize=(12, 13))
    order = sorted(df['Chamber'].unique())
    sns.boxplot(data=df, x='Chamber', y='Depo_THK', hue='Precursor', order=order, ax=ax[0, 0])
    ax[0, 0].set_title('챔버별 증착 두께 (Å)')
    sns.boxplot(data=df, x='Chamber', y='Particle', hue='Precursor', order=order, ax=ax[0, 1], showfliers=False)
    ax[0, 1].set_title('챔버별 파티클 (이상치 제외 표시)')
    sns.boxplot(data=df, x='Precursor', y='Depo_Rate', ax=ax[1, 0])
    ax[1, 0].set_title('가스별 증착 속도 (Å/s) : TEOS 가 약 2배 빠름')
    sns.histplot(data=df, x='Uniformity', hue='Precursor', bins=40, ax=ax[1, 1])
    ax[1, 1].set_title('두께 균일도 (EDGE-CENTER 차 / 두께, %)')
    rate = df.groupby('Recipe_Dev_Count')[['Depo_THK_y', 'Particle_y']].mean() * 100
    rate.plot(kind='bar', ax=ax[2, 0], color=['#3182bd', '#e6550d'], rot=0)
    ax[2, 0].set_title('레시피 이탈 개수별 불량률 (%)')
    ax[2, 0].set_xlabel('표준 레시피에서 벗어난 파라미터 수')
    lot = df.groupby('LOT')['Depo_THK_y'].sum()
    ax[2, 1].hist(lot[lot > 0], bins=25, color='#756bb1')
    ax[2, 1].set_title('두께 불량이 있는 LOT 의 LOT당 불량 수')
    ax[2, 1].set_xlabel('LOT 내 불량 웨이퍼 수 (최대 25)')
    fig.tight_layout()
    return fig


# %% 4. 분류 모델 (FDC) - 후보·하이퍼파라미터는 강사님 설정 그대로
def get_classifiers(w):
    cw = {0: 1, 1: w}
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=0.5, class_weight=cw, random_state=42),
        'KNN': KNeighborsClassifier(),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=w, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(class_weight=cw, random_state=42, verbosity=-1, n_estimators=100,
                                   learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                   importance_type='gain'),
    }


def binary_model(df, target, split='random'):
    X = make_features(df)
    y = df[target + '_y']
    feature_cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = split_data(X, y, df['LOT'], split, stratify=y)
    w = round((y_train == 0).sum() / max((y_train == 1).sum(), 1))   # 양품/불량 비율로 가중치

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    for name, model in get_classifiers(w).items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results.append({'Model명': name, 'model': model, 'scaler': scaler,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, zero_division=0),
                        'Recall': recall_score(y_test, y_pred, zero_division=0),
                        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                        'y_pred': y_pred, 'feature_cols': feature_cols, 'split': split})
    # [공정] F1 동점이면 Recall(불량 놓치지 않기) 우선 → 불량 유출이 가장 큰 손실
    df_res = pd.DataFrame(results).sort_values(['F1-Score', 'Recall'], ascending=False).reset_index(drop=True)

    n = len(df_res)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n))
    for i in range(n):
        r = df_res.loc[i]
        cm = confusion_matrix(y_test, r['y_pred'], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, 0],
                    xticklabels=['정상(0)', '불량(1)'], yticklabels=['정상(0)', '불량(1)'])
        axes[i, 0].set_title(f"{r['Model명']}  F1={r['F1-Score']:.3f}")
        axes[i, 0].set_xlabel('예측'); axes[i, 0].set_ylabel('실제')
        axes[i, 1].bar(['놓친 불량(FN)', '과검(FP)'], [cm[1, 0], cm[0, 1]], color=['#d62728', '#ff9896'])
        axes[i, 1].set_title('불량 유출 vs 과검')
    fig.tight_layout()

    joblib.dump(df_res.loc[0], os.path.join(MODEL_DIR, f'binary_{target}.pkl'))
    return fig, df_res[['Model명', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]


def _load_or_train(df, kind, target):
    path = os.path.join(MODEL_DIR, f'{kind}_{target}.pkl')
    if not os.path.exists(path):          # 모델이 없으면(서버 재시작 등) 자동 학습
        (binary_model if kind == 'binary' else linear_model)(df, target)
        plt.close('all')
    return joblib.load(path)


def _importance(model, cols):
    if hasattr(model, 'feature_importances_'):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, 'coef_'):
        imp = np.abs(np.ravel(model.coef_))
    else:
        imp = np.full(len(cols), np.nan)
    if np.nansum(imp) > 0:
        imp = imp / np.nansum(imp)
    return (pd.DataFrame({'Feature': cols, 'Importance': imp})
            .sort_values('Importance', ascending=False).reset_index(drop=True))


def binary_predict(df, target):
    r = _load_or_train(df, 'binary', target)
    X_s = r['scaler'].transform(make_features(df, r['feature_cols']))
    y_pred = r['model'].predict(X_s)
    cm = pd.DataFrame(confusion_matrix(df[target + '_y'], y_pred, labels=[0, 1]),
                      index=['실제 정상', '실제 불량'], columns=['예측 정상', '예측 불량'])
    df_anormal = df[y_pred == 1]
    return r['Model명'], cm, df_anormal, _importance(r['model'], r['feature_cols'])


# %% 5. 회귀 모델 (VM) - 후보·하이퍼파라미터는 강사님 설정 그대로
def get_regressors():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge(L2)': Ridge(alpha=10.0),
        'Lasso(L1)': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5,
                                                       random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=10, verbosity=-1, random_state=42),
    }


def linear_model(df, target, split='random'):
    X = make_features(df)
    y = df[target]
    feature_cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = split_data(X, y, df['LOT'], split)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    for name, model in get_regressors().items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results.append({'Model명': name, 'model': model, 'scaler': scaler,
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'R2 Score': r2_score(y_test, y_pred),
                        'y_test': y_test, 'y_pred': y_pred, 'feature_cols': feature_cols, 'split': split})
    df_res = pd.DataFrame(results).sort_values('R2 Score', ascending=False).reset_index(drop=True)

    n = len(df_res)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(12, 3.6 * ((n + 2) // 3)))
    axes = np.array(axes).reshape(-1)
    lo, hi = y_test.min(), y_test.max()
    for ax, (_, r) in zip(axes, df_res.iterrows()):
        ax.scatter(r['y_test'], r['y_pred'], s=6, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
        ax.set_title(f"{r['Model명']}  R2={r['R2 Score']:.3f}", fontsize=9)
        ax.set_xlabel('실제'); ax.set_ylabel('예측'); ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()

    joblib.dump(df_res.loc[0], os.path.join(MODEL_DIR, f'linear_{target}.pkl'))
    return fig, df_res[['Model명', 'MAE', 'RMSE', 'R2 Score']]


def vm_predict(df, target):
    """저장된 VM 모델로 '계측 전' 예측값"""
    r = _load_or_train(df, 'linear', target)
    return r['model'].predict(r['scaler'].transform(make_features(df, r['feature_cols'])))


def linear_predict(df, target, sigma):
    """VM 예측값과 실측값의 잔차가 ±σ 를 벗어나면 '설비 상태 이상' 의심"""
    y_pred = vm_predict(df, target)
    res = df[target] - y_pred
    up, lo = res.mean() + sigma * res.std(), res.mean() - sigma * res.std()
    out = df[['LOT', 'WF', 'Chamber', 'Precursor', 'Recipe_Dev_Count']].copy()
    out['Actual'], out['Predicted'], out['Residual'] = df[target], y_pred, res
    out['Is_Outlier'] = ((res > up) | (res < lo)).astype(int)
    return out[out['Is_Outlier'] == 1]


def model_summary():
    """[공정] 저장된 4개 채택 모델 요약 (2×2)"""
    rows = []
    for kind, task in [('binary', 'FDC (분류)'), ('linear', 'VM (회귀)')]:
        for t in TARGETS:
            p = os.path.join(MODEL_DIR, f'{kind}_{t}.pkl')
            if os.path.exists(p):
                r = joblib.load(p)
                score = f"F1 {r['F1-Score']:.3f}" if kind == 'binary' else f"R² {r['R2 Score']:.3f}"
                rows.append({'구분': task, 'Target': t, '채택 모델': r['Model명'], '점수': score,
                             '분할 방식': 'LOT 단위' if r.get('split') == 'lot' else '무작위'})
            else:
                rows.append({'구분': task, 'Target': t, '채택 모델': '(미학습)', '점수': '-', '분할 방식': '-'})
    return pd.DataFrame(rows)


# %% 6. [공정] 실시간 FDC 판정 규칙
def recipe_interlock(row):
    """투입 전 레시피 점검 : 표준 레시피에서 벗어난 파라미터 목록"""
    bad = [f"{c}={row[c]} (표준 {sp})" for c, sp in RECIPE_SETPOINT.items()
           if abs(row[c] - sp) > sp * RECIPE_TOL]
    std_t = PROCESS_TIME_STD.get(row['Precursor'])
    if std_t is not None and row['Main process time'] != std_t:
        bad.append(f"Main process time={row['Main process time']} (표준 {std_t})")
    return bad


def spc_rules(values, mean, std, one_sided=False):
    """SPC 판정 (Western Electric 룰 일부)
       R1 : 1점이 3σ 밖            → 인터락(설비 정지)
       R2 : 연속 3점 중 2점이 2σ 밖 → 경고
       R3 : 연속 7점이 평균 한쪽     → 경고 (공정 쏠림/드리프트)"""
    v = np.asarray(values, dtype=float)
    z = (v - mean) / std
    last = z[-1]
    if (last > 3) or (not one_sided and last < -3):
        return 'R1', '3σ 이탈 → 인터락'
    if len(z) >= 3:
        z3 = z[-3:]
        if (z3 > 2).sum() >= 2 or (not one_sided and (z3 < -2).sum() >= 2):
            return 'R2', '연속 3점 중 2점 2σ 이탈 → 경고'
    if len(z) >= 7:
        z7 = z[-7:]
        prev = z[-8] if len(z) >= 8 else 0.0      # 쏠림이 '처음' 7점에 도달한 순간에만 1번 경고
        if ((z7 > 0).all() and prev <= 0) or (not one_sided and (z7 < 0).all() and prev >= 0):
            return 'R3', '7점 연속 평균 한쪽 → 쏠림 경고'
    return None, ''
