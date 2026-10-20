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

st.set_page_config(page_title="PECVD VM·FDC 대시보드", page_icon="🔬", layout="wide")

# ============================== 화면 스타일 ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input { font-family: 'Noto Sans KR', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1 { font-size: 1.7rem !important; font-weight: 700 !important; }
h2, h3 { font-weight: 700 !important; }
[data-testid="stMetric"] {
    background: rgba(128,128,128,.06); border: 1px solid rgba(128,128,128,.22); border-radius: 12px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] p { opacity: .75; font-size: .85rem; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
.page-desc { opacity: .78; font-size: .95rem; margin: -6px 0 18px 0; }
.flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 10px 0; }
.flow span.step { background: #eef4fc; color: #1d5fae; border: 1px solid #cfe0f5; border-radius: 999px;
    padding: 4px 12px; font-size: .85rem; font-weight: 500; }
.flow span.arrow { opacity: .6; }
.badge { display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: .8rem; font-weight: 700; }
.b-ok { background: #e3f4ec; color: #0f6b47; } .b-warn { background: #fdf1d8; color: #8a5a00; }
.b-stop { background: #fde3e3; color: #a02626; } .b-info { background: #eef4fc; color: #1d5fae; }
.status-box { border-radius: 12px; padding: 14px 18px; margin: 6px 0 12px 0; font-weight: 500; }
.s-run { background: #eef4fc; border: 1px solid #cfe0f5; color: #1d5fae; }
.s-stop { background: #fde3e3; border: 1px solid #f5bcbc; color: #a02626; }
.s-done { background: #e3f4ec; border: 1px solid #b9e2cc; color: #0f6b47; }
.s-idle { background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25); }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_csv_dataset():
    df = ai.load_csv()
    return df, df[ai.PROCESS_COLS + ai.TARGETS]

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


# ============================== 공통 위젯 ==============================
TARGET_LABEL = {'Depo_THK': '증착 두께 (Depo_THK)', 'Particle': '파티클 (Particle)'}


def header(title, desc):
    st.title(title)
    st.markdown(f"<div class='page-desc'>{desc}</div>", unsafe_allow_html=True)


def pick_target(key):
    return st.segmented_control("Target", ai.TARGETS, default='Depo_THK', key=key,
                                format_func=lambda t: TARGET_LABEL[t]) or 'Depo_THK'


def fmt_table(df, best_col=None, pct_cols=(), dec=3):
    fmt = {c: f'{{:.{dec}f}}' for c in df.select_dtypes('float').columns}
    sty = df.style.format(fmt)
    if best_col is not None:
        sty = sty.apply(lambda r: ['background-color:#eef4fc;font-weight:600' if r.name == 0 else '' for _ in r], axis=1)
    return sty


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================== 0. 요약 ==============================
def page_summary():
    header("PECVD 공정 VM · FDC 대시보드",
           "증착 전·중 공정 데이터만으로 <b>불량을 판정(FDC)</b>하고 <b>두께·파티클을 예측(VM)</b>합니다.")
    st.markdown("""<div class='flow'>
      <span class='step'>① 입고 · LOT/WF</span><span class='arrow'>→</span>
      <span class='step'>② 전처리 · 세정/식각/애싱/CMP</span><span class='arrow'>→</span>
      <span class='step'>③ 증착 전 계측 · PRE</span><span class='arrow'>→</span>
      <span class='step'>④ 메인 증착 · 챔버/가스/RF/시간</span><span class='arrow'>→</span>
      <span class='step' style='background:#fdf1d8;color:#8a5a00;border-color:#f3dca6'>⑤ 증착 후 계측 · POST (예측 대상)</span>
    </div>""", unsafe_allow_html=True)

    ok = df_ai['Recipe_Dev_Count'] == 0
    c = st.columns(4)
    c[0].metric("웨이퍼 / LOT", f"{len(df_ai):,} / {df_ai['LOT'].nunique()}")
    c[1].metric("두께 불량률 (양측 3σ)", f"{df_ai['Depo_THK_y'].mean():.2%}", f"{df_ai['Depo_THK_y'].sum()}장", delta_color="off")
    c[2].metric("파티클 불량률 (상측 3σ)", f"{df_ai['Particle_y'].mean():.2%}", f"{df_ai['Particle_y'].sum()}장", delta_color="off")
    c[3].metric("레시피 이탈 시 두께 불량률", f"{df_ai.loc[~ok, 'Depo_THK_y'].mean():.1%}",
                f"정상 레시피 {df_ai.loc[ok, 'Depo_THK_y'].mean():.1%}", delta_color="off")

    st.subheader("채택 모델")
    summ = ai.model_summary()
    cols = st.columns(2)
    for col, (task, label) in zip(cols, [('FDC (분류)', 'FDC · 불량 판정'), ('VM (회귀)', 'VM · 값 예측')]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                sub = summ[summ['구분'] == task]
                cc = st.columns(2)
                for k, (_, r) in enumerate(sub.iterrows()):
                    cc[k].metric(TARGET_LABEL[r['Target']], r['채택 모델'].replace('Classifier', '').replace('Regressor', ''),
                                 r['점수'], delta_color="off")

    a, b = st.columns([1, 2])
    with a:
        if st.button("4개 모델 한 번에 학습", type="primary", use_container_width=True):
            with st.spinner("FDC 2개 + VM 2개 모델 학습 중..."):
                for t in ai.TARGETS:
                    ai.binary_model(df_ai, t); plt.close('all')
                    ai.linear_model(df_ai, t); plt.close('all')
            st.session_state['model_ver'] = time.time()
            st.rerun()
    with b:
        with st.container(border=True):
            st.markdown("""
**이 대시보드가 고려한 공정 조건**
- 계측 후 값(POST·EDGE CENTER)과 식별자(LOT·WF·투입 시각)는 **입력에서 제외** — 데이터 누수 방지
- **표준 레시피 이탈량 · 이탈 개수**, **RF 반사율**, **CMP 연마량**을 공정 변수로 추가
- 공정 시간은 **가스별 표준(SiH4 120s / TEOS 60s)** 기준으로 판단
- 실시간 FDC = **투입 전 레시피 인터락 + 증착 후 SPC 3개 룰**
""")


# ============================== 1. 데이터 ==============================
def page_data():
    header("데이터 검색", "컬럼을 검색하고 챔버·가스 등 공정 조건별로 묶어서 확인합니다.")
    q = st.text_input("컬럼 검색", placeholder="예) THK, PC, RF, dev")
    cols = [w for w in df_ai.columns if q.lower() in w.lower()]
    if not cols:
        st.warning("일치하는 컬럼이 없습니다.")
        return
    a, b = st.columns([1, 1])
    chk = a.selectbox("컬럼 선택", cols)
    group = b.multiselect("그룹 기준", ['Chamber', 'Precursor', 'Recipe_Dev_Count', 'LOT'])
    if pd.api.types.is_numeric_dtype(df_ai[chk]):
        d = df_ai[chk].describe()
        c = st.columns(5)
        for k, (lab, key) in enumerate([('평균', 'mean'), ('표준편차', 'std'), ('최소', 'min'), ('중앙값', '50%'), ('최대', 'max')]):
            c[k].metric(lab, f"{d[key]:,.2f}")
        if group:
            st.dataframe(df_ai.groupby(group)[chk].agg(['count', 'mean', 'std', 'min', 'max']).style.format('{:,.2f}'),
                         use_container_width=True)
    else:
        st.dataframe(df_ai[chk].value_counts().rename('웨이퍼 수'), use_container_width=True)
    show_cols = st.multiselect("표로 볼 컬럼", df_ai.columns.tolist(),
                               default=['LOT', 'WF', 'Chamber', 'Precursor', chk] if chk not in ['LOT', 'WF', 'Chamber', 'Precursor'] else ['LOT', 'WF', 'Chamber', 'Precursor'])
    st.dataframe(df_ai[show_cols], use_container_width=True, height=380)


# ============================== 2~4. EDA ==============================
DEFAULT_EDA = ['pre clean chemical flow rate', 'ashing time', 'RF Power', 'Main process time',
               'RF reflect Power Max', 'PRE PC', 'Depo_THK', 'Particle']


def _pick_vars(key):
    outs = ai.outlier_cols(df_number)
    sel = st.multiselect("변수 선택", df_number.columns.tolist(), default=DEFAULT_EDA, key=key,
                         help="빨간 제목(●) = 평균 ± 3σ 밖 값이 있는 변수")
    st.caption(f"3σ 밖 값이 있는 변수 {len(outs)}개 : " + ", ".join(outs))
    return sel


def page_hist():
    header("히스토그램", "변수별 분포와 평균(실선) · ±3σ(빨간 점선). 파란 음영이 관리 범위입니다.")
    sel = _pick_vars("h")
    if sel:
        show(ai.hist_pro(df_number[sel]))


def page_line():
    header("Line Chart (Trend)", "웨이퍼 투입 순서대로 본 추세. 빨간 점 = ±3σ 밖 웨이퍼.")
    sel = _pick_vars("l")
    if sel:
        show(ai.line_pro(df_number[sel]))


def page_corr():
    header("Target 과 상관계수", "공정 변수(레시피 이탈 변수 포함)와 Target 의 선형 상관. 선형 관계만 보여주므로 트리 모델의 중요도와는 다를 수 있습니다.")
    target = pick_target("t4")
    fig_bar, fig_hm, _ = ai.corr_pro(df_ai, target)
    a, b = st.columns([1, 1])
    with a:
        show(fig_bar)
    with b:
        show(fig_hm)


# ============================== 5. 공정 조건 ==============================
def page_process():
    header("공정 조건 분석", f"표준 레시피(설정값 ±{ai.RECIPE_TOL:.0%}) 이탈과 챔버 · 가스 조건이 품질에 주는 영향")
    tab1, tab2, tab3, tab4 = st.tabs(["레시피 이탈", "챔버 · 가스", "품질 지표", "LOT 분포"])
    with tab1:
        show(ai.recipe_rate_fig(df_ai))
        st.dataframe(ai.recipe_check(df_ai), hide_index=True, use_container_width=True)
        st.info("공정 시간 이탈 → 두께 불량 급증, 애싱·세정 이탈 → 파티클 불량 급증. 공정 시간은 가스별 표준(SiH4 120s, TEOS 60s) 기준.")
    with tab2:
        a, b = st.columns([1, 2])
        with a:
            st.markdown("**챔버 × 가스 운영 현황 (웨이퍼 수)**")
            st.dataframe(pd.crosstab(df_ai['Chamber'], df_ai['Precursor'], margins=True, margins_name='합계'),
                         use_container_width=True)
            st.caption("A·B = SiH4 전용, C·D = TEOS 전용 → 챔버 차이와 가스 차이가 겹쳐 있음")
        with b:
            show(ai.chamber_fig(df_ai))
    with tab3:
        show(ai.quality_fig(df_ai))
    with tab4:
        show(ai.lot_fig(df_ai))
        st.caption("두께 불량은 특정 LOT 에 몰려 있음 → LOT 단위로 레시피가 바뀐 경우. 레시피 인터락으로 투입 전 차단 가능.")


# ============================== 6~7. 모델 ==============================
def page_fdc():
    header("분류 모델 (FDC)", "증착 전·중 데이터로 불량 여부를 판정합니다. F1 → Recall 순으로 1위를 채택합니다(불량 유출 최소화).")
    target = pick_target("t6")
    with st.spinner("분류 모델 5개 학습 중..."):
        fig, res = ai.binary_model(df_ai, target)
    st.session_state['model_ver'] = time.time()
    best = res.iloc[0]
    c = st.columns(4)
    c[0].metric("채택 모델", best['Model명'].replace('Classifier', ''))
    c[1].metric("F1-Score", f"{best['F1-Score']:.3f}")
    c[2].metric("Recall (불량 검출률)", f"{best['Recall']:.3f}")
    c[3].metric("놓친 불량 / 과검", f"{best['놓친 불량(FN)']} / {best['과검(FP)']}")
    t1, t2, t3 = st.tabs(["모델 비교", "혼동 행렬", "지표 설명"])
    with t1:
        x, y = st.columns([3, 2])
        x.dataframe(fmt_table(res, best_col=True), hide_index=True, use_container_width=True)
        with y:
            show(ai.score_fig(res, 'F1-Score'))
    with t2:
        show(fig)
    with t3:
        st.markdown("""
| 지표 | 의미 | 공정에서의 의미 |
|---|---|---|
| **Recall** | 실제 불량 중 모델이 불량으로 맞춘 비율 | 낮으면 불량이 후속 공정으로 **유출** |
| **Precision** | 예측한 불량 중 실제 불량인 비율 | 낮으면 멀쩡한 웨이퍼로 설비가 **불필요하게 정지** |
| **F1-Score** | Precision·Recall 조화평균 | 두 손실의 균형 |
| Accuracy | 전체 중 맞춘 비율 | 불량이 3% 라 전부 정상이라 해도 97% → 참고용 |
""")


def page_vm():
    header("회귀 모델 (VM)", "계측기 없이 공정 조건만으로 두께 · 파티클 값을 예측합니다. R² 1위를 채택합니다.")
    target = pick_target("t7")
    with st.spinner("회귀 모델 7개 학습 중..."):
        fig, res = ai.linear_model(df_ai, target)
    st.session_state['model_ver'] = time.time()
    best = res.iloc[0]
    unit = 'Å' if target == 'Depo_THK' else '개'
    c = st.columns(4)
    c[0].metric("채택 모델", best['Model명'].replace('Regressor', ''))
    c[1].metric("R²", f"{best['R2 Score']:.3f}")
    c[2].metric("평균 오차 (MAE)", f"{best['MAE']:.1f} {unit}")
    c[3].metric("RMSE", f"{best['RMSE']:.1f} {unit}")
    t1, t2 = st.tabs(["모델 비교", "실제 vs 예측"])
    with t1:
        x, y = st.columns([3, 2])
        x.dataframe(fmt_table(res, best_col=True), hide_index=True, use_container_width=True)
        with y:
            show(ai.score_fig(res, 'R2 Score'))
        st.caption("MAE = 평균 절대 오차 · RMSE = 큰 오차에 민감한 오차 · R² = 1 에 가까울수록 잘 맞춤")
    with t2:
        st.caption("빨간 점선(y = x) 에 가까울수록 정확. ★ = 채택 모델")
        show(fig)


# ============================== 8~9. 판정 ==============================
def page_judge():
    header("유/불 확인", "저장된 FDC 모델로 전체 웨이퍼를 판정하고, 판정 근거(중요 변수)를 확인합니다.")
    target = pick_target("t8")
    name, cm, df_anormal, imp = ai.binary_predict(df_ai, target)
    c = st.columns(4)
    c[0].metric("모델", name.replace('Classifier', ''))
    c[1].metric("판정 불량", f"{len(df_anormal)} 장", f"{len(df_anormal)/len(df_ai):.2%}", delta_color="off")
    c[2].metric("실제 불량 중 검출", f"{cm.iloc[1, 1]} / {cm.iloc[1].sum()}")
    c[3].metric("과검", f"{cm.iloc[0, 1]} 장")
    a, b = st.columns([1, 2])
    with a:
        st.markdown("**혼동 행렬**")
        st.dataframe(cm, use_container_width=True)
    with b:
        st.markdown("**중요 변수 Top 10**")
        st.bar_chart(imp.head(10).set_index('Feature'), horizontal=True, color=ai.C_BLUE, height=300)
    st.markdown("**불량 판정 웨이퍼**")
    st.dataframe(df_anormal[['LOT', 'WF', 'Chamber', 'Precursor', 'Recipe_Dev_Count', target] + ai.PROCESS_COLS],
                 hide_index=True, use_container_width=True, height=320)


def page_residual():
    header("VM 잔차 이상 감지",
           "VM 예측(공정 조건상 나와야 할 값)과 실측값의 차이가 크면 → 레시피로 설명되지 않는 <b>설비 이상</b> 의심")
    a, b = st.columns([1, 1])
    with a:
        target = pick_target("t9")
    with b:
        sigma = st.slider('잔차 기준 : 평균 ± (sigma × 표준편차)', 1.0, 6.0, 3.0, 0.5)
    out = ai.linear_predict(df_ai, target, sigma)
    c = st.columns(3)
    c[0].metric("이상 웨이퍼", f"{len(out)} 장", f"{len(out)/len(df_ai):.2%}", delta_color="off")
    by = out['Chamber'].value_counts().reindex(sorted(df_ai['Chamber'].unique()), fill_value=0)
    c[1].metric("가장 많은 챔버", by.idxmax() if len(out) else '-', f"{by.max()} 장" if len(out) else '', delta_color="off")
    c[2].metric("레시피 정상인데 이상", f"{(out['Recipe_Dev_Count'] == 0).sum()} 장",
                help="레시피는 표준대로였는데 결과가 예측과 다름 → 설비 상태 점검 대상")
    st.dataframe(out.style.format({'Actual': '{:.1f}', 'Predicted': '{:.1f}', 'Residual': '{:+.1f}'}),
                 hide_index=True, use_container_width=True, height=380)


# ============================== 10. 실시간 FDC ==============================
def page_realtime():
    header("실시간 FDC", "웨이퍼가 들어올 때마다 <b>① 투입 전 레시피 점검</b> → <b>② 증착 후 SPC 판정</b>. 3σ 이탈 시 설비 정지(인터락).")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            target = pick_target("t10")
        with c2:
            mode = st.segmented_control("감시 값", ['실측값 (계측 후)', 'VM 예측값 (계측 전)'],
                                        default='실측값 (계측 후)', key='m10') or '실측값 (계측 후)'
        with c3:
            speed = st.select_slider("속도", options=['느림', '보통', '빠름'], value='보통')
        use_recipe = st.toggle("레시피 이탈 웨이퍼는 투입 전 정지 (레시피 인터락)", value=False)
    delay = {'느림': 0.1, '보통': 0.03, '빠름': 0.0}[speed]

    ss = st.session_state
    cfg = (target, mode, use_recipe)
    if ss.get('cfg') != cfg:
        ss.update(cfg=cfg, run=False, lot_id=0, idx=0, chart=None, events=[], paused_reason='', done=False)

    mean, std, ucl, lcl = ai.spec_limits(df_ai, target)
    one_sided = lcl is None
    lots = df_ai['LOT'].unique().tolist()
    series = get_vm_series(target, ss.get('model_ver', 0)) if mode.startswith('VM') else df_ai[target]

    b1, b2, b3, _ = st.columns([1, 1, 1.6, 2])
    if b1.button("▶ 시작", type="primary", use_container_width=True):
        ss.update(run=True, lot_id=0, idx=0, chart=None, events=[], paused_reason='', done=False)
        st.rerun()
    if b2.button("⏸ 일시정지" if ss.run else "⏵ 재개", use_container_width=True, disabled=bool(ss.paused_reason)):
        ss.run = not ss.run
        st.rerun()
    if b3.button("✔ 확인 후 다음 웨이퍼", disabled=not ss.paused_reason, use_container_width=True):
        ss.update(run=True, paused_reason='', idx=ss.idx + 1)
        st.rerun()

    status = st.empty()
    holder = st.empty()
    kpi = st.container()

    def set_status():
        lot_txt = f"LOT {min(ss.lot_id + 1, len(lots))} / {len(lots)}"
        if ss.paused_reason:
            status.markdown(f"<div class='status-box s-stop'>⛔ 설비 정지 · {ss.paused_reason}</div>", unsafe_allow_html=True)
        elif ss.get('done'):
            status.markdown("<div class='status-box s-done'>✅ 전체 LOT 모니터링 완료</div>", unsafe_allow_html=True)
        elif ss.run:
            status.markdown(f"<div class='status-box s-run'>● 모니터링 중 · {lot_txt}</div>", unsafe_allow_html=True)
        else:
            status.markdown(f"<div class='status-box s-idle'>대기 · {lot_txt} · Mean {mean:.1f} · UCL {ucl:.1f}"
                            + ("" if one_sided else f" · LCL {lcl:.1f}") + "</div>", unsafe_allow_html=True)

    def draw(lot, xs, ys, marks):
        fig, ax = plt.subplots(figsize=(12, 3.4))
        lo_band = mean - 2 * std if not one_sided else mean - 5 * std
        ax.axhspan(mean - 3 * std if not one_sided else mean - 5 * std, mean + 3 * std, color=ai.C_BLUE, alpha=0.05, lw=0)
        ax.axhspan(lo_band, mean + 2 * std, color=ai.C_BLUE, alpha=0.05, lw=0)
        ax.axhline(mean, color=ai.INK2, lw=1)
        for k, colr, lab in [(3, ai.C_RED, '3σ'), (2, ai.C_WARN, '2σ')]:
            ax.axhline(mean + k * std, color=colr, ls='--', lw=1)
            ax.text(25.2, mean + k * std, f'+{lab}', va='center', fontsize=8, color=colr)
            if not one_sided:
                ax.axhline(mean - k * std, color=colr, ls='--', lw=1)
                ax.text(25.2, mean - k * std, f'−{lab}', va='center', fontsize=8, color=colr)
        ax.plot(np.array(xs) + 1, ys, color=ai.C_BLUE, lw=1.2, marker='o', ms=4)
        style = {'R1': ('X', ai.C_RED, '3σ 이탈(인터락)'), 'R2': ('o', ai.C_WARN, '2σ 2/3점(경고)'),
                 'R3': ('s', ai.C_VIOLET, '7점 쏠림(경고)'), 'RECIPE': ('^', ai.INK, '레시피 이탈')}
        used = set()
        for x, y, kind in marks:
            m, colr, lab = style[kind]
            ax.plot(x + 1, y, marker=m, ms=11, color=colr, ls='none', mec='white', mew=1.5,
                    label=lab if kind not in used else None)
            used.add(kind)
        if used:
            ax.legend(loc='upper left', ncol=4)
        ax.set_xlim(0.5, 26); ax.set_xticks(range(1, 26))
        ax.set_ylim(mean - 5 * std, mean + 5 * std)
        ax.set_title(f"{lot}   ·   {TARGET_LABEL[target]}   ·   {mode}", loc='left')
        ax.set_xlabel("Wafer No.")
        ax.grid(axis='x', visible=False)
        fig.tight_layout()
        holder.pyplot(fig, use_container_width=True)
        plt.close(fig)

    def show_kpi():
        with kpi:
            ev = pd.DataFrame(ss.events) if ss.events else pd.DataFrame(columns=['판정'])
            k = st.columns(4)
            k[0].metric("⛔ R1 인터락", int((ev['판정'] == 'R1').sum()))
            k[1].metric("⚠ R2 경고", int((ev['판정'] == 'R2').sum()))
            k[2].metric("↗ R3 쏠림", int((ev['판정'] == 'R3').sum()))
            k[3].metric("▲ 레시피 이탈", int((ev['판정'] == '레시피 이탈').sum()))
            if len(ev):
                st.dataframe(ev.iloc[::-1], hide_index=True, use_container_width=True, height=260)

    set_status()
    if ss.run:
        lot = lots[ss.lot_id]
        d = df_ai[df_ai['LOT'] == lot]
        vals = series.loc[d.index].tolist()
        marks = [m for m in (ss.chart or {}).get('marks', []) if (ss.chart or {}).get('lot') == lot]
        for i in range(ss.idx, len(d)):
            ss.idx = i
            row = d.iloc[i]
            bad = ai.recipe_interlock(row)
            if bad:
                marks.append((i, vals[i], 'RECIPE'))
                ss.events.append({'LOT': lot, 'WF': row['WF'], 'Chamber': row['Chamber'], '판정': '레시피 이탈',
                                  '내용': ', '.join(bad), '감시값': round(vals[i], 1)})
                if use_recipe:
                    ss.chart = dict(lot=lot, xs=list(range(i + 1)), ys=vals[:i + 1], marks=marks)
                    ss.update(run=False, paused_reason=f"레시피 인터락 · {lot} WF{row['WF']} · " + ', '.join(bad))
                    send_discord(f"⛔ [레시피 인터락] {lot} WF{row['WF']} {row['Chamber']} : {', '.join(bad)}")
                    st.rerun()
            rule, msg = ai.spc_rules(vals[:i + 1], mean, std, one_sided)
            if rule:
                marks.append((i, vals[i], rule))
                ss.events.append({'LOT': lot, 'WF': row['WF'], 'Chamber': row['Chamber'], '판정': rule,
                                  '내용': msg, '감시값': round(vals[i], 1)})
            ss.chart = dict(lot=lot, xs=list(range(i + 1)), ys=vals[:i + 1], marks=marks)
            draw(lot, list(range(i + 1)), vals[:i + 1], marks)
            if rule == 'R1':
                ss.update(run=False, paused_reason=f"{lot} WF{row['WF']} ({row['Chamber']}) · {msg} · 값 {vals[i]:.1f}")
                send_discord(f"🚨 [FDC 인터락] {lot} WF{row['WF']} {row['Chamber']} | {target}({mode})"
                             f" = {vals[i]:.1f} (UCL {ucl:.1f})")
                st.rerun()
            time.sleep(delay)
        ss.lot_id += 1
        ss.idx = 0
        if ss.lot_id < len(lots):
            st.rerun()
        ss.update(run=False, done=True)
        st.rerun()
    elif ss.chart:
        c = ss.chart
        draw(c['lot'], c['xs'], c['ys'], c['marks'])
    else:
        draw(lots[0], [], [], [])
    show_kpi()


# ============================== 메뉴 ==============================
def main():
    st.sidebar.markdown("### 🔬 PECVD VM · FDC")
    st.sidebar.caption("Project3 · 가상 계측 / 이상 감지")
    pages = {
        "개요": [st.Page(page_summary, title="요약", icon=":material/dashboard:", url_path="summary", default=True),
                 st.Page(page_data, title="데이터 검색", icon=":material/search:", url_path="data")],
        "탐색 (EDA)": [st.Page(page_hist, title="히스토그램", icon=":material/bar_chart:", url_path="hist"),
                       st.Page(page_line, title="Line Chart", icon=":material/show_chart:", url_path="line"),
                       st.Page(page_corr, title="상관계수", icon=":material/grid_on:", url_path="corr"),
                       st.Page(page_process, title="공정 조건 분석", icon=":material/precision_manufacturing:", url_path="process")],
        "모델": [st.Page(page_fdc, title="분류 모델 (FDC)", icon=":material/rule:", url_path="fdc"),
                 st.Page(page_vm, title="회귀 모델 (VM)", icon=":material/trending_up:", url_path="vm")],
        "판정 · 모니터링": [st.Page(page_judge, title="유/불 확인", icon=":material/fact_check:", url_path="judge"),
                          st.Page(page_residual, title="VM 잔차 이상 감지", icon=":material/troubleshoot:", url_path="residual"),
                          st.Page(page_realtime, title="실시간 FDC", icon=":material/monitor_heart:", url_path="realtime")],
    }
    st.navigation(pages).run()
    st.sidebar.divider()
    st.sidebar.caption("모델이 없으면 요약 화면의 **4개 모델 한 번에 학습**을 먼저 누르세요.")


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
