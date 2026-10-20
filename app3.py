import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.web import cli as stcli

import project3_service as ai

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
# 실행 : streamlit run app3.py   (Spyder 에서 이 파일 실행해도 자동 실행)

st.set_page_config(page_title="PECVD VM·FDC 대시보드", layout="wide")


@st.cache_data
def get_csv_dataset():
    df = ai.load_csv()
    df_number = df[ai.PROCESS_COLS + ai.TARGETS]
    return df, df_number

df_ai, df_number = get_csv_dataset()


@st.cache_data
def get_vm_series(target, _version):
    return pd.Series(ai.vm_predict(df_ai, target), index=df_ai.index)


def send_discord(msg):
    """웹훅 주소는 코드에 쓰지 않고 Streamlit Secrets(DISCORD_WEBHOOK_URL)에서 읽음"""
    try:
        url = st.secrets.get("DISCORD_WEBHOOK_URL", "")
    except Exception:
        url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        return False
    try:
        return requests.post(url, json={"content": msg}, timeout=3).status_code == 204
    except Exception:
        return False


TARGET_LABEL = {'Depo_THK': 'Depo_THK (증착 두께)', 'Particle': 'Particle (파티클)'}
def pick_target(key):
    return st.radio("Target을 선택 하세요:", ai.TARGETS, horizontal=True, index=0,
                    format_func=lambda t: TARGET_LABEL[t], key=key)


def pick_split(key):
    s = st.radio("평가 데이터 분할 방식:", ['무작위 8:2 (강사님 방식)', 'LOT 단위 8:2 (처음 보는 LOT 로 평가)'],
                 horizontal=True, key=key)
    return 'lot' if s.startswith('LOT') else 'random'


menus = ["0. 요약",
         "1. Data 검색",
         "2. 히스토그램",
         "3. Line Chart",
         "4. Target과 상관계수",
         "5. 공정 조건 분석",
         "6. 분류 모델 (FDC)",
         "7. 회귀 모델 (VM)",
         "8. 유/불 확인",
         "9. VM 잔차 이상 감지",
         "10. 실시간 FDC"]


def main():
    menu = st.sidebar.radio("Project3 : PECVD 가상 계측(VM) · 이상 감지(FDC)", menus)
    st.sidebar.caption("모델이 없으면 6·7번 메뉴(또는 0번 '4개 모델 학습')를 먼저 실행하세요.")

    # ------------------------------------------------------------------ 0. 요약
    if menu == menus[0]:
        st.title("PECVD 공정 VM · FDC 대시보드")
        c = st.columns(5)
        c[0].metric("웨이퍼 / LOT", f"{len(df_ai):,} / {df_ai['LOT'].nunique()}")
        c[1].metric("두께 불량률 (양측 3σ)", f"{df_ai['Depo_THK_y'].mean():.2%}")
        c[2].metric("파티클 불량률 (상측 3σ)", f"{df_ai['Particle_y'].mean():.2%}")
        c[3].metric("레시피 이탈 웨이퍼", f"{(df_ai['Recipe_Dev_Count'] > 0).mean():.1%}")
        ok = df_ai['Recipe_Dev_Count'] == 0
        c[4].metric("레시피 이탈 시 두께 불량률", f"{df_ai.loc[~ok, 'Depo_THK_y'].mean():.1%}",
                    f"정상 레시피 {df_ai.loc[ok, 'Depo_THK_y'].mean():.1%}", delta_color="off")

        st.subheader("채택 모델 (FDC · VM × 두께 · 파티클)")
        st.dataframe(ai.model_summary(), hide_index=True, use_container_width=True)
        split = pick_split("split0")
        if st.button("4개 모델 한 번에 학습", type="primary"):
            with st.spinner("FDC 2개 + VM 2개 모델 학습 중..."):
                for t in ai.TARGETS:
                    ai.binary_model(df_ai, t, split); plt.close('all')
                    ai.linear_model(df_ai, t, split); plt.close('all')
            st.session_state['model_ver'] = time.time()
            st.rerun()

        st.markdown("""
**공정 흐름과 데이터**
`입고(LOT·WF)` → `전처리(세정·식각·애싱·CMP)` → `증착 전 계측(PRE)` → `메인 증착(챔버·가스·RF·시간)` → `증착 후 계측(POST)`

- **FDC** : 증착 전/중 데이터만으로 불량을 판정 → 인터락으로 후속 공정 유출 차단
- **VM** : 계측기 없이 두께·파티클을 예측 → 전수 검사 효과
- 계측 후 값(POST THK · POST PC · EDGE CENTER)과 식별자(LOT · WF · TKIN time)는 **입력에서 제외**(데이터 누수 방지)
- 표준 레시피 대비 **이탈량 · 이탈 개수 · RF 반사율 · CMP 연마량**을 공정 변수로 추가
""")

    # ------------------------------------------------------------------ 1. 검색
    elif menu == menus[1]:
        search_char = st.text_input("1. 찾는 문자를 입력하세요 :", placeholder="예) THK, PC, RF, dev")
        cols = [w for w in df_ai.columns if search_char.lower() in w.lower()]
        if not cols:
            st.warning("일치하는 컬럼이 없습니다.")
            return
        chk_col = st.radio("2. 컬럼를 선택하세요:", cols, horizontal=True, index=0)
        temp_li = df_ai[chk_col].unique().tolist()
        st.write(f'{temp_li[:20]}... {len(temp_li)}개' if len(temp_li) > 20 else f'{temp_li}  {len(temp_li)}개')
        if pd.api.types.is_numeric_dtype(df_ai[chk_col]):
            st.dataframe(df_ai[chk_col].describe().to_frame().T)
        group = st.multiselect("그룹 기준 (예: Chamber, Precursor) :", ['Chamber', 'Precursor', 'LOT', 'Recipe_Dev_Count'])
        if group and pd.api.types.is_numeric_dtype(df_ai[chk_col]):
            st.dataframe(df_ai.groupby(group)[chk_col].agg(['count', 'mean', 'std', 'min', 'max']))

        st.subheader("공정 변수 체크박스 필터링")
        selected = []
        cc = st.columns(4)
        for i, f in enumerate(df_ai.columns):
            with cc[i % 4]:
                if st.checkbox(f, value=False, key=f"chk_{f}"):
                    selected.append(f)
        if selected:
            st.dataframe(df_ai[selected])

    # ------------------------------------------------------------------ 2~4. EDA
    elif menu == menus[2]:
        st.caption("빨간 제목 = 평균 ± 3σ 밖의 값이 있는 변수")
        st.pyplot(ai.hist_pro(df_number))

    elif menu == menus[3]:
        st.caption("웨이퍼 투입 순서대로 본 Trend (빨간 점선 = ±3σ)")
        st.pyplot(ai.line_pro(df_number))

    elif menu == menus[4]:
        target = pick_target("t4")
        fig, _ = ai.corr_pro(df_ai, target)
        st.pyplot(fig)

    # ------------------------------------------------------------------ 5. 공정 조건
    elif menu == menus[5]:
        st.subheader("레시피(Setpoint) 이탈과 불량률")
        st.caption(f"설정값 대비 ±{ai.RECIPE_TOL:.0%} 를 넘으면 이탈 / 공정 시간은 가스별 표준(SiH4 120s, TEOS 60s) 기준")
        st.dataframe(ai.recipe_check(df_ai), hide_index=True, use_container_width=True)
        st.subheader("챔버 × 가스 운영 현황")
        st.dataframe(pd.crosstab(df_ai['Chamber'], df_ai['Precursor'], margins=True))
        st.info("Chamber A·B 는 SiH4 전용, C·D 는 TEOS 전용으로 운영 → 챔버 차이와 가스 차이가 겹쳐 있음")
        st.pyplot(ai.process_pro(df_ai))

    # ------------------------------------------------------------------ 6. 분류
    elif menu == menus[6]:
        target = pick_target("t6")
        split = pick_split("s6")
        st.dataframe(pd.DataFrame([["전체 중 맞춘 비율", "예측한 불량 중 실제 불량인 비율",
                                    "실제 불량 중 모델이 불량으로 맞춘 비율", "Precision·Recall 조화평균"]],
                                  columns=["Accuracy", "Precision", "Recall", "F1-Score"]), hide_index=True)
        with st.spinner("분류 모델 5개 학습 중..."):
            fig, res = ai.binary_model(df_ai, target, split)
        st.session_state['model_ver'] = time.time()
        st.success(f"채택 : {res.iloc[0]['Model명']}  (F1 → Recall 순으로 선정, 불량 유출 최소화)")
        st.dataframe(res, hide_index=True)
        st.pyplot(fig)

    # ------------------------------------------------------------------ 7. 회귀
    elif menu == menus[7]:
        target = pick_target("t7")
        split = pick_split("s7")
        st.markdown('###### MAE : 평균 절대 오차 · RMSE : 큰 오차에 민감한 오차 · R² : 1 에 가까울수록 잘 맞춤')
        with st.spinner("회귀 모델 7개 학습 중..."):
            fig, res = ai.linear_model(df_ai, target, split)
        st.session_state['model_ver'] = time.time()
        st.success(f"채택 : {res.iloc[0]['Model명']}")
        st.dataframe(res, hide_index=True)
        st.pyplot(fig)

    # ------------------------------------------------------------------ 8. 유/불 확인
    elif menu == menus[8]:
        target = pick_target("t8")
        name, cm, df_anormal, imp = ai.binary_predict(df_ai, target)
        st.subheader(f"{name} 판정 불량 : {len(df_anormal)}/{len(df_ai)}  ({len(df_anormal)/len(df_ai):.2%})")
        c1, c2 = st.columns([1, 2])
        c1.dataframe(cm)
        c2.bar_chart(imp.head(10).set_index('Feature'))
        st.dataframe(df_anormal[['LOT', 'WF', 'Chamber', 'Precursor', 'Recipe_Dev_Count', target] + ai.PROCESS_COLS])

    # ------------------------------------------------------------------ 9. 잔차
    elif menu == menus[9]:
        target = pick_target("t9")
        st.caption("VM 예측(공정 조건상 나와야 할 값)과 실측값이 크게 다르면 → 레시피로 설명 안 되는 설비 이상 의심")
        sigma = st.slider('잔차 기준 : 평균 ± (sigma × 표준편차)', 1.0, 6.0, 3.0, 0.5)
        out = ai.linear_predict(df_ai, target, sigma)
        st.subheader(f"이상 웨이퍼 : {len(out)}/{len(df_ai)}  ({len(out)/len(df_ai):.2%})")
        st.dataframe(out.groupby('Chamber').size().rename('이상 수').to_frame().T)
        st.dataframe(out)

    # ------------------------------------------------------------------ 10. 실시간 FDC
    elif menu == menus[10]:
        realtime_fdc()


def realtime_fdc():
    st.markdown("##### 10. 실시간 FDC : 투입 전 레시피 인터락 + 증착 후 SPC 판정")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        target = pick_target("t10")
    with c2:
        mode = st.radio("감시 값 :", ['실측값 (계측 후)', 'VM 예측값 (계측 전)'], horizontal=True)
    with c3:
        speed = st.select_slider("속도", options=[0.0, 0.02, 0.05, 0.1], value=0.02)
    use_recipe = st.checkbox("레시피 이탈 웨이퍼도 투입 전 정지 (레시피 인터락)", value=False)

    ss = st.session_state
    cfg = (target, mode, use_recipe)
    if ss.get('cfg') != cfg:   # 조건이 바뀌면 처음부터
        ss.update(cfg=cfg, run=False, lot_id=0, idx=0, chart=None, events=[], paused_reason='')

    mean, std, ucl, lcl = ai.spec_limits(df_ai, target)
    one_sided = lcl is None
    lots = df_ai['LOT'].unique().tolist()
    series = get_vm_series(target, ss.get('model_ver', 0)) if mode.startswith('VM') else df_ai[target]

    b1, b2, b3 = st.columns(3)
    if b1.button("▶ 시작", type="primary"):
        ss.update(run=True, lot_id=0, idx=0, chart=None, events=[], paused_reason='')
        st.rerun()
    if b2.button("⏸ 정지" if ss.run else "⏵ 재개"):
        ss.run = not ss.run
        st.rerun()
    if b3.button("확인 → 다음 웨이퍼 진행", disabled=not ss.paused_reason):
        ss.update(run=True, paused_reason='', idx=ss.idx + 1)
        st.rerun()

    st.caption(f"LOT {min(ss.lot_id + 1, len(lots))}/{len(lots)}  |  Mean {mean:.1f}  UCL {ucl:.1f}"
               + ("" if one_sided else f"  LCL {lcl:.1f}") + "  |  R1: 3σ 이탈=인터락, R2: 2σ 2/3점=경고, R3: 7점 쏠림=경고")
    holder = st.empty()

    def draw(lot, xs, ys, marks):
        fig, ax = plt.subplots(figsize=(11, 3.2))
        for k, colr in [(3, 'r'), (2, 'orange')]:
            ax.axhline(mean + k * std, color=colr, ls='--', lw=1)
            if not one_sided:
                ax.axhline(mean - k * std, color=colr, ls='--', lw=1)
        ax.axhline(mean, color='y', ls='--', lw=1)
        ax.plot(xs, ys, color='#1f77b4', lw=0.8, marker='o', ms=3)
        for x, y, kind in marks:
            ax.plot(x, y, marker={'R1': 'X', 'R2': 'o', 'R3': 's', 'RECIPE': '^'}[kind], ms=10,
                    color={'R1': 'red', 'R2': 'orange', 'R3': 'purple', 'RECIPE': 'black'}[kind])
        ax.set_xlim(-0.5, 25); ax.set_ylim(mean - 5 * std, mean + 5 * std)
        ax.set_title(f"{lot}  [{target} / {mode}]"); ax.set_xlabel("wafer"); ax.grid(alpha=0.3)
        holder.pyplot(fig); plt.close(fig)

    if ss.run:
        lot = lots[ss.lot_id]
        d = df_ai[df_ai['LOT'] == lot]
        vals = series.loc[d.index].tolist()
        marks = [m for m in (ss.chart or {}).get('marks', []) if (ss.chart or {}).get('lot') == lot]
        for i in range(ss.idx, len(d)):
            ss.idx = i
            row = d.iloc[i]
            # (1) 투입 전 : 레시피 점검
            bad = ai.recipe_interlock(row)
            if bad:
                marks.append((i, vals[i], 'RECIPE'))
                ss.events.append({'LOT': lot, 'WF': row['WF'], 'Chamber': row['Chamber'], '판정': '레시피 이탈',
                                  '내용': ', '.join(bad), '감시값': round(vals[i], 1)})
                if use_recipe:
                    ss.chart = dict(lot=lot, xs=list(range(i + 1)), ys=vals[:i + 1], marks=marks)
                    draw(lot, list(range(i + 1)), vals[:i + 1], marks)
                    ss.update(run=False, paused_reason='레시피 인터락 : ' + ', '.join(bad))
                    send_discord(f"⛔ [레시피 인터락] {lot} WF{row['WF']} {row['Chamber']} : {', '.join(bad)}")
                    st.rerun()
            # (2) 증착 후 : SPC 판정
            rule, msg = ai.spc_rules(vals[:i + 1], mean, std, one_sided)
            if rule:
                marks.append((i, vals[i], rule))
                ss.events.append({'LOT': lot, 'WF': row['WF'], 'Chamber': row['Chamber'], '판정': rule,
                                  '내용': msg, '감시값': round(vals[i], 1)})
            draw(lot, list(range(i + 1)), vals[:i + 1], marks)
            ss.chart = dict(lot=lot, xs=list(range(i + 1)), ys=vals[:i + 1], marks=marks)
            if rule == 'R1':
                ss.update(run=False, paused_reason=f"{lot} WF{row['WF']} {msg}")
                send_discord(f"🚨 [FDC 인터락] {lot} WF{row['WF']} {row['Chamber']} | {target}({mode})"
                             f" = {vals[i]:.1f} (UCL {ucl:.1f})")
                st.rerun()
            time.sleep(speed)
        ss.lot_id += 1
        ss.idx = 0
        if ss.lot_id < len(lots):
            st.rerun()
        ss.run = False
        st.success("전체 LOT 모니터링 완료")
    elif ss.chart:
        c = ss.chart
        draw(c['lot'], c['xs'], c['ys'], c['marks'])

    if ss.paused_reason:
        st.error("⛔ 설비 정지 : " + ss.paused_reason + "  → 확인 후 '다음 웨이퍼 진행'")
    if ss.events:
        ev = pd.DataFrame(ss.events)
        k = st.columns(4)
        k[0].metric("R1 인터락", int((ev['판정'] == 'R1').sum()))
        k[1].metric("R2 경고", int((ev['판정'] == 'R2').sum()))
        k[2].metric("R3 쏠림", int((ev['판정'] == 'R3').sum()))
        k[3].metric("레시피 이탈", int((ev['판정'] == '레시피 이탈').sum()))
        st.dataframe(ev.iloc[::-1], hide_index=True, use_container_width=True)


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
