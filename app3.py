import streamlit as st

import project3_service  as ai
from streamlit.web import cli as stcli
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
pd.options.mode.chained_assignment = None
# from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
import time
import requests

# pip install scikit-learn


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




st.set_page_config(page_title="대시보드", layout="wide")
# [★ 핵심] 저장 시마다 외부 파일을 강제로 새로고침하여 메모리에 올립니다.
# importlib.reload(ai)

@st.cache_data
def get_csv_dataset():
    df=ai.load_csv()
    df_number = df.select_dtypes(include=['number'])
    return df, df_number

# 데이터 가져오기 (이미 로드되었다면 0.1초만에 리턴됨)

df_ai, df_number=get_csv_dataset()


def send_discord(msg):
    

    webhook_url = "https://discord.com/api/webhooks/1525755958708670624/yC4rrqVdDZP8K6DuUdrjkFnwWRohDl38reTerV3xJcdw_fIL1Fql9xFb1xkiQIix4aCO"
    data = {
        "content": msg,
       
    }
    
    response = requests.post(webhook_url, json=data)
    
    if response.status_code == 204:
        print("메시지 전송 성공!")
    else:
        print(f"실패 코드: {response.status_code}")
           
menus=  [ "1.AI Data 검색",   
   "2. 히스토그램",  
   "3. Line Chart", 
   "4. Target과 상관계수", 
 
   "5. 분류 모델",  
   "6. 회귀 모델",
   "7. 유/불 확인",
   "8. 예상 target 값",
   "9. 실시간 FDC"
   ]

def main():
  
    # 사이드바 메뉴   
    menu = st.sidebar.radio(
        "Project3 : 가상 계측(VM) 및 이상 감지(FDC)",
        menus
    )
    if menu == menus[0] :
        search_char=st.text_input("1.찾는 문자를 입력하세요 :", placeholder="예)setpoint, CTC_chA_WaferNo=='1' ")        
        words=df_ai.columns
        cols=[w for w in words if search_char.lower() in w.lower()] 
        chk_col = st.radio("2. 컬럼를 선택하세요:", cols, horizontal=True,index=0)
        #   unique()
        temp_li=df_ai[chk_col].unique().tolist()         
        if len(temp_li) > 20 :
            result = f'{temp_li[:20]}... {len(temp_li)}개'
        else :
            result=f'{temp_li}  {len(temp_li)}개'
            
        st.write(result)
        temp_type= df_ai[chk_col].dtype
        st.write(df_ai[chk_col].head())
        if (temp_type == 'int64') or (temp_type == 'float64') :
            df=df_ai.select_dtypes(include=[np.number])   
            temp_df=pd.DataFrame([[]])
            temp_df['mean']=df[chk_col].mean()
            temp_df['min']=df[chk_col].min()
            temp_df['max']=df[chk_col].max()
            temp_df['std']=df[chk_col].std()
        selected_list = st.multiselect("웨이퍼를 선택하세요: 선택 내용이 없을시에는 모두 프린트 합니다 ", cols)
        if st.button("함수 실행"): 
            if (len(selected_list) != 0) & (chk_col not in selected_list) :                 
                gsr=df_ai.groupby(selected_list)[chk_col].value_counts()
                
                st.dataframe(gsr)
                
        
        st.title("공정 변수 체크박스 필터링")
        
        all_features = df_ai.columns.tolist()
        selected_features = []
        
        # 1. 화면 가로축을 4개의 열(Column)로 분할
        cols = st.columns(4)
        
        # 2. 반복문을 돌며 각 열에 체크박스를 순차적으로 배치
        for i, feature in enumerate(all_features):
            # i % 4 를 통해 0, 1, 2, 3번 열에 번갈아가며 배정
            with cols[i % 4]:
                # 기본적으로 모든 변수가 체크된 상태(value=True)로 시작
                if st.checkbox(feature, value=False, key=f"chk_{feature}"):
                    selected_features.append(feature)
        
        st.markdown("---")
        
        # 3. 결과 출력
        if selected_features:
            st.subheader("📊 선택된 변수 데이터 확인")
            st.dataframe(df_ai[selected_features])
        else:
            st.error("🚨 선택된 변수가 없습니다. 체크박스를 선택해 주세요.")
            
       
    elif menu == menus[1]:     
       
         fig = ai.hist_pro(df_number)
      
         st.pyplot(fig)
         
    elif menu == menus[2] :    
      
         fig = ai.line_pro(df_number)
         st.pyplot(fig)     
         
        
    elif menu == menus[3]:  
        st.session_state.clear()
        df = df_ai.select_dtypes(include='number')
        target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0)
        fig, results = ai.corr_pro(df_ai, target)
        # st.dataframe(results)
        st.pyplot(fig)
  
    elif menu == menus[4]: 
        st.session_state.clear()
        target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0) 
        #################
        head_01=["전체 데이터 중 맞게 맞춘 비율", "예측한 불량 중 실제 불량인 비율", 
                 "실제 불량 중 모델이 불량으로 맞춘 비율","정밀도(Precision)와 재현율(Recall)의 조화평균값"]
        #####################
        # F1_score = 2 * (precision * recall)/(precision + recall)
        head_df = pd.DataFrame([list(head_01)], columns=["Accuracy","Precision","Recall","F1-Score"])    
               
        st.dataframe(head_df)        
        fig, results = ai.binary_model(df_ai, target)         
        st.dataframe(results)
        st.pyplot(fig)
         
    elif menu == menus[5]: 
         st.session_state.clear()
         target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0)
         
         head_01 = ['평균절대오차:<br> 오차(틀린 양)의 절대값을 평균',
                    '평균제곱근오차:<br> 오차를 제곱해서 평균을 낸 뒤, 다시 루트',
                    '모델 예측 비율']
         
         head_df = pd.DataFrame([head_01], 
                                   columns=[ "MAE", "RMSE","R2 Score"] )
          
        
         # [★ 핵심] 데이터프레임을 HTML로 변환한 후, HTML 렌더링을 허용하여 출력합니다.
         # escape=False 설정을 해주어야 <br> 태그가 깨지지 않고 줄바꿈으로 작동합니다.
         st.markdown('###### '+head_df.to_html(escape=False, index=False), unsafe_allow_html=True)
         
         fig, results = ai.linear_model(df_ai,target)         
         st.dataframe(results)
         st.pyplot(fig)
         
         
    elif menu == menus[6]: 
        st.session_state.clear()
        target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0)
        name, cm, df_anormal, importance_df=ai.binary_predict(df_ai, target)
        st.subheader(f'{name} 불량 :{len(df_anormal)}/{len(df_ai)} ->  불량률{len(df_anormal)/len(df_ai)}')
        st.dataframe(cm)
        st.dataframe(df_anormal)
        st.dataframe(importance_df)
                     
         
  
         
    elif menu == menus[7]: 
         st.session_state.clear()
         target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0)
         t_sigma= st.slider(    
            'mean +- (sigma * std)',            
            max_value=6.0,
            min_value=0.0,    
            value=1.5,
            step=0.5
        ) 
         df_linear=ai.linear_predict(df_ai, target, t_sigma)    
         st.subheader(f'불량 :{len(df_linear)}/{len(df_ai)} ->  불량률{len(df_linear)/len(df_ai)}')
        
         st.dataframe(df_linear)
         
    elif menu == menus[8]:
        st.markdown("##### " + menu)
        
        target_col = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0)
     
        
        #  target_col을 수정 하여 한개의 공정 에 따라서 fdc를 판단 할 수 도 있음
        # target_col = "Depo_THK" 
        
        # 1. 세션 상태 초기화
        if "run" not in st.session_state:
            st.session_state.run = False
        if "disabled" not in st.session_state:
            st.session_state.disabled = True
        if "stop_name" not in st.session_state:
            st.session_state.stop_name = "Stop"
        if "lot_id" not in st.session_state:
            st.session_state.lot_id = 0
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0
        if "stat_str" not in st.session_state:
            st.session_state.stat_str = []
         
            
        
        # ★ 추가: 이상치 발생 시 차트 상태를 보존하기 위한 세션 변수
        if "chart_data" not in st.session_state:
            st.session_state.chart_data = None
        if "dfdata" not in st.session_state:
            st.session_state["dfdata"]= 'ok'
            st.session_state["col"]= df_ai.columns
            st.session_state["errlist"]=[]
            st.session_state["lots"] = df_ai["LOT"].unique().tolist()          
            st.session_state['std']= df_ai[target_col].std()
            st.session_state['mean']= df_ai[target_col].mean()
            st.session_state['ucl']= df_ai[target_col].mean()+(3*df_ai[target_col].std())
            st.session_state['lcl']= df_ai[target_col].mean()-(3*df_ai[target_col].std())
         
          
           
            
            
          
            
        
        speed = 0.03
        
        # 2. 버튼 영역
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            start_button = st.button("차트 그리기 시작")
            if start_button:
                st.session_state.run = True
                st.session_state.current_index = 0
                st.session_state.disabled = True
                st.session_state.stop_name = "Stop"
                st.session_state.chart_data = None  # 차트 초기화
                st.session_state["errlist"]=[]
                
                 
                st.session_state.lot_id = 0
                st.session_state.stat_str = []
                
                st.session_state["dfdata"]= None
                
                
                st.rerun()
                
                
                
                # 1. 세션 상태 초기화
               
                
               
                
                
                
                
                
                
                
                
        
        with col2:
            stop_button = st.button(st.session_state.stop_name, type="secondary")
            if stop_button:
                if st.session_state.stop_name == "Stop":
                    st.session_state.stop_name = "Restart"
                    st.session_state.run = False
                else:
                    st.session_state.stop_name = "Stop"
                    st.session_state.run = True
                st.rerun()
        
        with col3:
            outlier_button = st.button(
                "이상치 발생", disabled=st.session_state.disabled, type="primary"
            )
            if outlier_button:
                st.session_state.disabled = True
                st.session_state.run = True
                # 다음 데이터부터 진행하기 위해 index 증가
                st.session_state.current_index += 1
                st.rerun()
        
        # 3. 차트 레이아웃 준비
        
       
        
        
        
        chart_placeholder = st.empty()
        
        
        # ★ 차트 그리기 헬퍼 함수
        def plot_chart(x_data, y_data, feat_mean, ucl,  lcl,feat_std,  xlen, lot_name, highlight_outlier=False,):
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axhline(y=feat_mean, color="y", linestyle="--", linewidth=1.5)
            ax.axhline(y=ucl, color="r", linestyle="--", linewidth=1.5)
            ax.axhline(y=lcl, color="r", linestyle="--", linewidth=1.5)
        
            ax.set_xlabel("wafer id")
            ax.set_title(f"{lot_name}")
            ax.set_ylabel("값 (Value)")
            ax.set_xlim(0, xlen)
            ax.set_ylim(feat_mean - (feat_std * 5), feat_mean + (feat_std * 5))
            ax.grid(True, linestyle="--", alpha=0.5)
        
            # 기본 점/선 그리기
            ax.plot(
                x_data,
                y_data,
                color="#1f77b4",
                linewidth=0.5,
                marker="o",
                mfc="#ff7f0e",
                mec="#ff7f0e",
                markersize=0.2,
            )
        
            # 이상치 지점 강조 표기
            if highlight_outlier:
                ax.plot(
                    x_data[-1],
                    y_data[-1],
                    marker="o",
                    markersize=6,
                    mfc="red",
                    mec="red",
                )
            
            return fig
        
        
        # 4. 실시간 실행 상태 처리
        if st.session_state.run:
            
            # st.session_state.lot_id 기준으로 
            lot_name = st.session_state["lots"][st.session_state.lot_id]
            df_t= df_ai[df_ai['LOT']==lot_name]            
            xlen=len(df_t)
            feat_list=df_t[target_col].to_list()
            st.session_state.stat_str=[]
            
            
            
            
            start_idx = st.session_state.current_index
        
            for i in range(start_idx, xlen):
                st.session_state.current_index = i
                x_data = list(range(i + 1))
                y_data = feat_list[: i + 1]
        
                last_value = y_data[-1]
                is_outlier = (last_value > st.session_state.ucl) or (last_value < st.session_state.lcl)
        
                # 차트 세션 데이터에 저장
                st.session_state.chart_data = {
                    "x_data": x_data,
                    "y_data": y_data,
                    "feat_mean": st.session_state['mean'],
                    "ucl": st.session_state['ucl'],
                    "lcl": st.session_state['lcl'],
                    "feat_std": st.session_state['std'],
                    "xlen": xlen,
                    "lot_name": lot_name,
                    "is_outlier": is_outlier,
                }
        
                # 화면 출력
                fig = plot_chart(
                    x_data,
                    y_data,
                    st.session_state['mean'],
                    st.session_state['ucl'],
                    st.session_state['lcl'],
                    st.session_state['std'],
                    xlen,
                    lot_name,
                    is_outlier,
                )
                chart_placeholder.pyplot(fig)
                plt.close(fig)
                
                
                
                
                
        
                # 이상치 탐지 시 멈춤 처리
                if is_outlier:
                    st.session_state["errlist"].append(df_t.iloc[i].tolist())
                    send_discord(str(df_t.iloc[i].tolist()))
                    st.session_state.run = False  # 정지
                    st.session_state.disabled = False  # 이상치 버튼 활성화
                    st.rerun()
        
                time.sleep(speed)
        
            # 1개 LOT 완료 처리
            if st.session_state.run:
                st.session_state.lot_id += 1
                st.session_state.current_index = 0
        
                if st.session_state.lot_id < len(st.session_state["lots"]):
                    st.rerun()
                else:
                    st.session_state.run = False
                    print("end")
                
        
        # 5. 멈춤 상태(이상치 발생 포함) 시 기존 차트 유지하여 그리기
        elif st.session_state.chart_data is not None:
            cd = st.session_state.chart_data
            fig = plot_chart(
                cd["x_data"],
                cd["y_data"],
                cd["feat_mean"],
                cd["ucl"],
                cd["lcl"],
                cd["feat_std"],
                cd["xlen"],
                cd["lot_name"],
                cd["is_outlier"],
            )
            chart_placeholder.pyplot(fig)
            plt.close(fig)
        
        print(st.session_state["col"])
        print(st.session_state["errlist"])
        
        
        if len(st.session_state["errlist"]) > 0 :  
        
                err_df = pd.DataFrame(
                    st.session_state["errlist"], 
                    columns=st.session_state["col"]
                )
                st.subheader(f'불량 row 수 {len(err_df)}')
               
                st.dataframe(err_df)
        
                     
 
      
        
        

    
    
    
    
        

#   streamlit run app.py

if __name__ == "__main__":
    
    # 현재 프로세스가 streamlit 환경 내에서 실행 중인지 확인
    if st.runtime.exists():
        main()
    else:
        # streamlit 환경이 아니면, 스스로를 streamlit으로 실행 (외부 터미널 효과)
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
