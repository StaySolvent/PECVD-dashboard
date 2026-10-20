# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:59:10 2026

@author: Admin
"""

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC   # model.feature_importances_  없음 
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import os
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
os.makedirs('model', exist_ok=True)   # [수정] model 폴더 없으면 저장 에러 → 자동 생성


#  pip install scikit-learn
#  pip install xgboost lightgbm

# Streamlit Cloud(Linux) 환경 나눔고딕 폰트 파일 경로
nanum_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(nanum_path):
    # 1. 서버(Linux) 환경: 폰트 파일 직접 등록
    fm.fontManager.addfont(nanum_path)
    plt.rc('font', family='NanumGothic')
else:
    # 2. 로컬 PC(Windows/Mac) 환경 예외 처리
    try:
        plt.rc('font', family='Malgun Gothic')  # Windows
    except:
        plt.rc('font', family='AppleGothic')   # Mac

# 음수 기호(-) 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# %%
def load_csv(CSV_PATH="data/semi_process_data2.csv"):           
    
    df = pd.read_csv(CSV_PATH, encoding="utf-8")        
    #1. 결측치 / 단일값 컬럼 삭제
    df = df.dropna(axis=1)
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() <= 1])       
    
    #2. Depo_THK, Particle  추가    
    df['Depo_THK']=df['POST THK AVG']-df['PRE THK AVG']  
    df['Particle']=df['POST PC']-df['PRE PC']
    df['Particle'] = df['Particle'].clip(lower=0)    # lower=0 옵션을 주면 0을 최하한선으로 고정하겠다는 의미입니다.
    # np.log1p(값은 0보다 커야 한다 )    
    ### 3*시그마로 유/불 ??????
    def labeling_pro(target):
       mean_val = df[target].mean()
       std_val = df[target].std()    
       # 3. 2시그마 경계값 설정 (상한선, 하한선)
       lower_limit = mean_val - 3 * std_val
       upper_limit = mean_val + 3 * std_val
       
       y = np.where((df[target] < lower_limit) | 
                              (df[target] > upper_limit), 1, 0)
       #우량=0(Negative), 불량=1(Positive)
       return y
   
    df['Depo_THK_y']=labeling_pro('Depo_THK')
    df['Particle_y']=labeling_pro('Particle')
    
    return df

'''
df_all=load_csv() 


'''  
    
# %%    

def hist_pro(df_number):
    count=1 
    wid=3 
    heig=len(df_number.columns)//wid + 1
    
  # 1. fig, axes 분해 및 count 초기화
    fig, axes = plt.subplots(heig, wid, figsize=(10, 20))
    count = 1 
    
    for col in df_number.columns:
        t_mean = df_number[col].mean()
        t_ucl = t_mean + (3 * df_number[col].std())
        t_lcl = t_mean - (3 * df_number[col].std())        
        if ((df_number[col].max() > t_ucl) |
            (df_number[col].min() < t_lcl)) :
            titlecolor = 'red'
            titlename = col + ': 이상치 확인'
        else :
            titlecolor = 'black'
            titlename = col 
            
        plt.subplot(heig, wid, count)
        count += 1 
        plt.title(titlename, fontsize=8, color=titlecolor)
        
        # 히스토그램 그리기
        plt.hist(df_number[col], color='skyblue', edgecolor='black')
        
        # (선택 사항) 계산해둔 평균과 3-Sigma(UCL, LCL) 수직선 추가
        plt.axvline(t_mean, color='red', linestyle='--', linewidth=1, label='Mean')
        plt.axvline(t_ucl, color='orange', linestyle=':', linewidth=1, label='UCL')
        plt.axvline(t_lcl, color='orange', linestyle=':', linewidth=1, label='LCL')
        
    plt.tight_layout()
    return fig


def line_pro(df_number):
    count=1 
    wid=3 
    heig=len(df_number.columns)//wid + 1
    # 1. fig, axes 분해 및 count 초기화
    fig, axes = plt.subplots(heig, wid, figsize=(10, 20))
    count = 1 
    for col in df_number.columns:
        t_mean = df_number[col].mean()
        t_ucl = t_mean + (3 * df_number[col].std())
        t_lcl = t_mean - (3 * df_number[col].std())        
        if ((df_number[col].max() > t_ucl) |
            (df_number[col].min() < t_lcl)) :
            titlecolor = 'red'
            titlename = col + ': 이상치 확인'
        else :
            titlecolor = 'black'
            titlename = col 
            
        plt.subplot(heig, wid, count)
        count += 1 
        plt.title(titlename, fontsize=8, color=titlecolor)
        
        plt.plot(range(len(df_number[col])), df_number[col])  
        plt.axhline(y=t_ucl , color='r', linestyle='dashed')
        plt.axhline(y=t_lcl , color='r', linestyle='dashed')
        
             
    
    plt.tight_layout()
    return fig

'''
line_pro(df)
hist_pro(df[['SiO2 Under SiN depo THK',
       'SiO2 Under SiN CMP THK', 'etch RF reflect power']])
'''

# %%
def corr_pro(df, target):
    # 1. 상관계수 행렬 계산 (절대값 적용)    
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 20))        
    tdf = df[df.columns[3:-2]].select_dtypes(include='number')   
    print(tdf.columns)
    
    # 두개의 타겟 간에 상관 관계가 명 확함
    # if target == 'Depo_THK':
    #     del tdf['Particle']
    # else :
    #     del tdf['Depo_THK']
        
        
    corr_matrix = tdf.corr()[target].abs()
    # threshold = 0.1
    # filtered_corr = corr_matrix[corr_matrix > threshold]
    sorted_cols = corr_matrix.sort_values(ascending=False).index
    # print(filtered_corr)
    
    
    # 삼각형 마스크 생성 (중복 정보 제거를 위해 윗부분을 가림)
    # mask = np.triu(np.ones_like(filtered_corr, dtype=bool))
    
    
    sns.heatmap(df[sorted_cols].corr(), 
                annot=True,           # 숫자 표시
                fmt=".2f",            # 소수점 둘째자리까지
                cmap='YlGnBu',        # 색상 테마 (Yellow-Green-Blue 혼합으로 부드러운 전개를 보여줌.)
                ax=ax[0], # 특정 서브플롯 위치 지정
                # mask=mask,            # 마스크 적용 (heatmap 일부  chart를 아보이게 한다 )
                linewidths=0.5,       # 칸 사이 구분선
                cbar_kws={"shrink": .8}) # 컬러바 크기 조절
   
    plt.title(f"공정 변수 간 상관관계 히트맵 (정렬 기준: {target})", fontsize=15, pad=20)
    # plt.show()      
   
    ax[1].plot(range(len(df)), tdf['Particle'], label='Particle' )
    ax[1].plot(range(len(df)), tdf['Depo_THK'], label='Depo_THK')
    ax[1].set_title('Particle and Depo_THK')
    ax[1].legend()
    plt.tight_layout()    
    return fig, sorted_cols[1:]   #맨 앞의 target 컬럼은 제외 한다


# %%  
def binary_model(df, target):   
    models = {
     # 1. 선형 및 확률 기반 모델
     "LogisticRegression": LogisticRegression(max_iter=1000, # 반복횟수
                                              C=0.5,  # 1 ~ 0 작을 수록 규제가 강해진다  
                                              class_weight={0: 1, 1: 32}, # 0: 우량(97%), 1(3%): 불량 비율 
                                              random_state=42),   # seed value
     
     # 2. 거리 및 커널 기반 모델
     # "SVM": SVC(probability=True, class_weight={0: 1, 1: 32}, random_state=42),
     "KNN": KNeighborsClassifier(),
     
     # 3. 앙상블 (배깅) : 
     # 복원 추출을 통해 여러 개의 서로 다른 샘플 데이터를 만들고(Bootstrap), 각각 학습시킨 뒤 결과를 하나로 합치는(Aggregating) 방법
     "RandomForestClassifier": RandomForestClassifier(n_estimators=100,    # 알상불 내부에 만들 결정 트리 개수 임
                                                      class_weight={0: 1, 1: 32},  # 1: 우량(97%), 0(3%): 불량 비율 
                                                      random_state=42),
     
     # 4. 앙상블 (부스팅) - 실무에서 가장 많이 쓰임
     # 이전 나무가 틀린 오차를 다음 나무가 순차적으로 보완해 나가는 '부스팅(Boosting)
     "XGBoost": XGBClassifier(
         scale_pos_weight=32,    # 0: 우량(97%), 1(3%): 불량 비율 
         eval_metric='logloss', 
         # logloss(교차 엔트로피 손실)는 모델이 예측한 확률이 실제 정답과 얼마나 다른지 패널티를 부여하며 계산합니다.
         # 0과 1을 분류하는 이진 분류(Binary Classification) 문제에서 가장 표준적이고 안정적인 평가지표
         random_state=42),
     
     
     "LightGBM":LGBMClassifier( 
         class_weight={0: 1, 1: 32},   # 1: 우량(97%), 0(3%): 불량 비율 
         random_state=42,
         verbosity=-1,               # 경고 메시지 출력 안 함
         n_estimators=100,           # 반복 횟수 (데이터가 적다면 조정) 오차를 보완해 나갈 순차적인 트리(부스팅 단계)의 개수
         learning_rate=0.05,         # 학습률 (조금 낮춰서 세밀하게 학습)
         num_leaves=15,              # 트리 복잡도 (데이터가 적으면 15~20으로 낮추기)
         min_child_samples=10,       # 리프 노드에 필요한 최소 데이터 (경고 방지용으로 조정)         
         importance_type='gain'      # 피처 중요도를 계산할 때 '이득(gain)' 기준 사용
         )
         # '''
         # 전체 데이터 수가 매우 많아서 3%에 해당하는 불량 데이터의 절대적인 개수가 수천 건 이상으로 충분하다면,
         # num_leaves=31,
         # min_child_samples=20,   # 데이터가 많으므로 최소 샘플 수도 늘려줌
         # '''
         }
        
    y=df[target+"_y"]    
    anormal = np.sum(y == 1)  # 불량
    # 5. 결과 확인
    print("\n--- 라벨링 결과 요약 ---")
    print(anormal, (anormal/len(df)*100), '%') # 비율 확인 (약 3.3%가 1로 나와야 함)
    
        
    # 꼭 지운다 !!!!!!
    ''' 
    ---  지워야 하는 feature
    Depo_THK: POST THK AVG, POST PC, Particle
    Particle: POST PC, POST THK AVG, Depo_THK
    
    ----- 살려두고 학습에 사용해야 하는 컬럼 (Features):
    PRE PC (공정 전 데이터이므로 사용 가능)
    PRE THK AVG (공정 전 데이터이므로 사용 가능)
    RF Power, Precursor flow rate, Chamber pressure 등의 순수 공정 센서 데이터
    ''' 
    
    
    cols= df.columns[:-4].tolist() 
    cols.remove('POST THK AVG')
    cols.remove('POST PC')  
    
    feature_cols = df[cols].select_dtypes(include='number').columns.tolist()
    print(feature_cols)
    
    #############
    # 1. 데이터 분할 (분류 모델이므로 stratify=y(유,불을 동일한 비율로)  권장)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. 스케일러 선언 및 적용 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Train 데이터로 fit & transform
    X_test_scaled = scaler.transform(X_test)        # Test 데이터는 transform만
    
    
    # 모델 평가 직전이나 직후에 아래 코드를 넣어 화면이나 터미널에 찍어보세요.
    print("실제 Test 데이터의 정상/불량 개수:", pd.Series(y_test).value_counts().to_dict())
########################    
    results=[]
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)     # 학습  
        y_pred = model.predict(X_test_scaled)  # 예측
        acc = accuracy_score(y_test, y_pred)     # 성능 지표 계산
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)        
        # 결과 저장
        results.append({
            "Model명": name,            "model" : model,
            "scaler" : scaler,            "Accuracy": acc,  
            "Precision": pre, 
            "Recall": rec,    
            "F1-Score": f1 , #정밀도(Precision)와 재현율(Recall)의 조화로운 평균값
            "y_pred" :  y_pred,
            "feature_cols" : feature_cols  })
        print(name, '예측불량', np.sum(y_pred == 1), '실제 분량',np.sum(y_test == 1) )   # 예측 불량
        # 3. 결과 테이블 생성 및 출력
        #F1-Score는 정밀도와 재현율을 결합한 지표입니다.
        #F1-Score는 1(또는 100%)에 가까울수록 모델의 성능이 좋다고 판단
    df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    df_results = df_results.reset_index(drop=True)
    print(df_results)
    '''    
    import matplotlib.pyplot as plt
    import seaborn as sns
    '''
    
    fig = plt.figure(figsize=(10,30))  # (width, height)
    count=1
    for row in range(len(df_results)) :    
        t_pred=df_results.loc[row]['y_pred']
        t_name=df_results.loc[row]['Model명']
   
        plt.subplot(6, 2, count)
        cm = confusion_matrix(y_test, t_pred) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fault (0)', 'Normal (1)'], 
                    yticklabels=['Fault (0)', 'Normal (1)'])
        plt.title('Confusion Matrix:'+t_name)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')      
        
        plt.subplot(6, 2, count+1) 
        plt.hist(x=y_test, alpha=0.5, color='b', label="실제"+str(np.sum(y_test == 0))+"/"+str(np.sum(y_test == 1)))
        plt.hist(x=y_pred+0.25, alpha=0.5, color='r', label="예측"+str(np.sum(t_pred == 0))+"/"+str(np.sum(t_pred == 1)))
        plt.xlabel('Actual Values (y_test)')
        plt.ylabel('Predicted Values (y_pred)')
        plt.title(t_name)
        plt.legend()
        plt.grid(True)        
        count +=2 
    plt.tight_layout()
    plt.show()
    
    # 파일로 저장   
    setmodel=df_results.loc[0]
    t_name=setmodel['Model명']
    print(t_name)
    # import joblib
    joblib.dump(setmodel, f'model/binary_{target}.pkl')
    # 확인: 첫 번째 모델로 예측해보기
    #first_model = loaded_results[0]['model']
    # prediction = first_model.predict(X_test)
    return fig, df_results[['Model명','Accuracy','Precision','Recall','F1-Score']]

'''
df_all.columns
binary_model(df_all, 'Depo_THK')
binary_model(df_all, 'Particle')
'''    
def binary_predict(df, target):
     results = joblib.load(f'model/binary_{target}.pkl')  #가장 효율성이 좋은 모델을 저장후 사용 
     model=results['model']  
     scaler=results['scaler']   
     name=results['Model명']
     feature_cols=results["feature_cols"]   #model 생성시 기준 columns
     # 1. 피처 중요도 추출 및 데이터프레임 변환
     importance_df = pd.DataFrame({
         'Feature': feature_cols,
         'Importance': model.feature_importances_
     })     
     # 2. 중요도 피처 내림차순 정렬
     importance_df = importance_df\
            .sort_values(by='Importance', ascending=False).reset_index(drop=True)   
     print(importance_df)
     
     y=df[target+"_y"]  
     X_scaled=scaler.transform(df[feature_cols])      
     
     y_pred=model.predict(X_scaled)
     
     df_anormal = df[(y_pred == 1)]  # 0: 우량, 1: 불량
     
     
     print(df_anormal.head())
     
     cm=confusion_matrix(y, y_pred) # 혼동 행렬
     print(cm, "===========") 
     return name, cm, df_anormal, importance_df   # [수정] 주석 해제 (app3.py 7번 메뉴 에러 방지)

'''
binary_predict(df_all, 'Depo_THK')
''' 

# %%
def linear_model(df, target):
    # 1. 회귀 모델 딕셔너리 생성   
    
    models = {
     # 1. 기본 선형 회귀 (기준점 설정용, 규제 없음)
     "LinearRegression": LinearRegression(),
     
     # 2. L2 규제 회귀 (모든 피처를 유지하되, 계수 크기를 줄여 과적합 방지)
     "Ridge(L2)": Ridge(
         alpha=10.0  # 데이터가 5,000개로 적은 편이므로 규제 강도를 살짝 높여(1.0 -> 10.0) 일반화 성능 확보
     ),
     
     # 3. L1 규제 회귀 (중요하지 않은 피처의 가중치를 0으로 만들어 자동 변수 선택)
     "Lasso(L1)": Lasso(
         alpha=1.0   # 피처가 너무 많을 때 효과적이며, 불필요한 노이즈 변수 제거에 도움
     ),
     
     # 4. 혼합 규제 (L1과 L2의 장점을 절반씩 결합)
     "ElasticNet": ElasticNet(
         alpha=1.0, 
         l1_ratio=0.5
     ),
     
     # 5. 배깅 앙상블 회귀 (이상치 노이즈에 강함)
     "RandomForestRegressor": RandomForestRegressor(
         n_estimators=100,
         max_depth=6,          # 5,000개 데이터에서 트리가 너무 깊어져 3% 이상치를 억지로 외우는 것 방지
         min_samples_leaf=5,   # 리프 노드에 최소 5개 이상의 샘플이 있도록 제한하여 오버피팅 차단
         random_state=42
     ),
     
     # 6. 부스팅 앙상블 회귀 (XGBoost)
     "XGBoost": XGBRegressor(
         n_estimators=100,
         learning_rate=0.05,   # 3%의 미세한 변화를 정밀하게 추적하기 위해 학습률 하향 (0.1 -> 0.05)
         max_depth=4,          # 과적합 방지를 위해 트리 깊이를 얇게 조정
         random_state=42
     ),
     
     # 7. 부스팅 앙상블 회귀 (LightGBM)
     "LightGBM": LGBMRegressor(
         n_estimators=100,
         learning_rate=0.05,   # 세밀한 보폭
         num_leaves=15,        # 5,000개 데이터 규모에 맞게 트리 복잡도를 31에서 15로 축소
         min_child_samples=10, # 적은 데이터 영역(3%)도 최소한의 규칙으로 반영될 수 있도록 설정
         verbosity=-1,
         random_state=42
     )}
   
    y=df[target] 
    
    # 꼭 지운다 
    '''    
    ---  지워야 하는 feature
    Depo_THK: POST THK AVG, POST PC, Particle
    Particle: POST PC, POST THK AVG, Depo_THK
    
    ----- 살려두고 학습에 사용해야 하는 컬럼 (Features):
    PRE PC (공정 전 데이터이므로 사용 가능)
    PRE THK AVG (공정 전 데이터이므로 사용 가능)
    RF Power, Precursor flow rate, Chamber pressure 등의 순수 공정 센서 데이터
    '''
    cols= df.columns[:-4].tolist() 
    cols.remove('POST THK AVG')
    cols.remove('POST PC') 
    
    feature_cols = df[cols].select_dtypes(include='number').columns.tolist()
   
   
    
    #########  model 전처리   
    # 1. 데이터 분할 
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], y, test_size=0.2, random_state=42)
    
    # 2. 스케일러 선언 및 적용
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Train 데이터로 fit & transform
    X_test_scaled = scaler.transform(X_test)        # Test 데이터는 transform만
    
    # 결과를 담을 리스트
    results = []
    
    # 3. 루프를 이용한 학습 및 예측 (수정된 부분)
    for name, model in models.items():
        # ★ 중요: 정규화된 데이터(X_train_scaled)로 학습해야 합니다.
        model.fit(X_train_scaled, y_train)
        
        # ★ 중요: 예측할 때도 정규화된 데이터(X_test_scaled)를 넣어줘야 합니다.
        y_pred = model.predict(X_test_scaled)
        
    # results 리스트에 append 하시면 됩니다.
        # 성능 지표 계산
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # 결과 저장
        results.append({
            "Model명": name,
            "model" : model,
            "scaler" : scaler,
            "MAE": mae,   #MAE (Mean Absolute Error, 평균 절대 오차)
            "RMSE": rmse, #RMSE (Root Mean Squared Error, 평균 제곱근 오차)
            "R2 Score": r2, #R2 Score (R-Squared, 결정 계수)
            "y_test": y_test,
            "y_pred": y_pred,
            "feature_cols" : feature_cols
        })
        
         
        
        
    # 4. 결과 테이블 생성 및 출력 (R2 Score가 높을수록 좋은 모델)
    df_results = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
    fig = plt.figure(figsize=(10,10))  # (width, height)
    count=1
    for row in range(len(df_results)) :
        plt.subplot(4, 3, count)
        count +=1
        # 산점도 그리기
        rows=df_results.iloc[row]
        
        plt.scatter(x=rows['y_test'], y=rows['y_pred'], alpha=0.5)
        
        # 기준선 (y=x) 그리기
        line_coords = [y_test.min(), y_test.max()]
        plt.plot(line_coords, line_coords, color='red', linestyle='--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values (y_test)')
        plt.ylabel('Predicted Values (y_pred)')
        plt.title(f"{target}:{rows['Model명']}[{rows['R2 Score']:.6f}]")
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    
    
    # plt.show()
    # 파일로 저장
 
    df_results = df_results.reset_index(drop=True)
    print(df_results)
    setmodel=df_results.loc[0]
    t_name=setmodel['Model명']
    print(t_name)
    joblib.dump(setmodel, f'model/linear_{target}.pkl')    
 
    
    return fig, df_results[['Model명','MAE','RMSE','R2 Score']]


'''
linear_model(df_all, 'Depo_THK')
linear_model(df_all, 'Particle')
'''
# %%

'''
linear_predict(df_all, 'Depo_THK', 1.5)
'''

def linear_predict(df, target, sigma):    
    # 저장된 파일 읽음
    results = joblib.load(f'model/linear_{target}.pkl')
    model=results['model']  
    scaler=results['scaler']   
    feature_cols=results["feature_cols"]   #model 생성시 기준 columns
    y_true=df[target]    
    
    X_scaled=scaler.transform(df[feature_cols])    # scaler 저장 되어 있는 방법으로 정규화 
    y_pred=model.predict(X_scaled)
   
    # 성능 지표 계산
    feature_cols.append(target)
    # 2. 잔차(오차) 계산
    residuals = y_true - y_pred
    # 3. 오차의 표준편차(Sigma) 계산
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
   
    # 4. 이상치 판단 임계치 설정 (1.5-Sigma 원칙 적용)
    threshold_upper = residual_mean + (sigma * residual_std)
    threshold_lower = residual_mean - (sigma * residual_std)
    print(threshold_upper, threshold_lower)
    # 5. 이상치 분류 (임계치를 벗어나면 0(false), 정상이면 1(true))
    # Particle의 경우 음수가 없으므로 upper 영역 위주로 필터링 가능
    is_outlier = ((residuals <= threshold_upper) & (residuals >= threshold_lower))
    
    df_result = pd.DataFrame({
        # 'Actual': y_true,   # df[target] 와 같음
        'Predicted': y_pred,
        'Residual': residuals,
        'Is_Outlier': is_outlier.astype(int) # True/False를 1/0으로 변환
    })
    
    df_result=pd.concat([df[feature_cols], df_result], axis=1)
    
    # 6. 이상치 데이터만 쏙 뽑아내기 (대시보드 출력용 df_anormal 생성)
    df_anormal = df_result[df_result['Is_Outlier'] == 0]
       
    return df_anormal
    
  
       
    
    
    
    
    
    
    
    
    

