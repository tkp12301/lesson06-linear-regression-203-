import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 연도별 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온에 선형회귀 직선을 맞춰 원하는 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 기준 기간 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만 연도 제외
    return yearly.reset_index(drop=True)


try:
    yearly = load_yearly_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 3:
    st.error("조건을 만족하는 연도가 너무 적어 회귀를 계산할 수 없습니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

# --- 직선을 만든 데이터 정보 ---
n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
st.info(
    f"회귀 직선에 사용한 해: **{n_years}개** · "
    f"시작 연도: **{start_year}년** · 끝 연도: **{end_year}년**"
)

# --- 연도 선택 & 예측 ---
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 상관계수 ---
c1, c2 = st.columns(2)
c1.metric("상관계수 (연도 vs 연평균기온)", f"{r:.3f}")
c2.metric("회귀 직선 기울기", f"{slope * 10:+.3f} °C / 10년")

# --- 그래프 ---
x_line = np.array([min(x.min(), year), max(x.max(), year)])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"], y=yearly["연평균기온"], mode="markers", name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line, y=slope * x_line + intercept, mode="lines", name="회귀 직선",
        line=dict(width=3, color="crimson"),
        hovertemplate="%{x:.0f}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year], y=[pred], mode="markers", name=f"{year}년 예상",
        marker=dict(size=14, color="gold", symbol="star", line=dict(width=1.5, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온과 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"회귀식: 기온 = {slope:.5f} × 연도 + {intercept:.3f} · "
    f"기준 기간 {LAST_YEAR}년까지, 관측일 {MIN_DAYS}일 미만인 해 제외. "
    "먼 미래나 과거 값은 직선을 늘려 본 단순 추정입니다."
)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(yearly.round(2), use_container_width=True, hide_index=True)
