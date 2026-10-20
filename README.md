# PECVD 가상 계측(VM) · 이상 감지(FDC) 대시보드

강사님 Project3 코드(후보 모델·하이퍼파라미터) 기반 + 공정 조건 반영

| 파일 | 역할 |
|---|---|
| app3.py | Streamlit 웹 화면 (메뉴 0~10) |
| project3_service.py | 데이터 처리 · 공정 변수 · 모델 학습/예측 · SPC 판정 |
| semi_process_data2.csv | 공정 데이터 (5,000 웨이퍼) |
| requirements.txt / packages.txt | Streamlit Cloud 설치 목록 (패키지 / 한글 폰트) |

실행 : `streamlit run app3.py`
디스코드 알림(선택) : Streamlit Cloud → Settings → Secrets 에 `DISCORD_WEBHOOK_URL = "..."`
