import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.font_manager as fm
from scipy.stats import ttest_ind

font_path = "fonts/Pretendard-SemiBold.ttf"

fm.fontManager.addfont(font_path)

font_name = fm.FontProperties(fname=font_path).get_name()

plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# 1. 데이터 로드 (Kaggle SECOM 데이터셋)
# secom.csv 에는 590개의 센서 데이터가 들어있음
data = pd.read_csv('uci-secom.csv')

X = data.copy()

# Time Date Month
data['Time'] = pd.to_datetime(data['Time'])
data['Date'] = data['Time'].dt.date
data['Month'] = data['Time'].dt.to_period('M')

# 결측치 비율이 20%를 넘는 열 제거
null_cols = X.columns[X.isnull().mean() > 0.2]
X.drop(columns=null_cols, inplace=True)

# 남은 결측치는 중앙값(Median)으로 단순 보정 수행
X.fillna(X.median(numeric_only=True), inplace=True)

# PF는 y로 빼줌
y = X['Pass/Fail']
# XGBoost는 0과 1의 타겟 라벨을 선호하므로 -1을 0으로 변환
y = y.replace(-1, 0)
X = X.drop(columns=['Time','Pass/Fail'])


# 여기부터
pass_data = X[y == 0]
fail_data = X[y == 1]

# Pass / Fail 각각의 표준편차
pass_std = pass_data.std()
fail_std = fail_data.std()

# Pass / Fail 각각의 표본 수
n_pass = pass_data.count()
n_fail = fail_data.count()

# Pooled Standard Deviation
pooled_std = np.sqrt(
    (
        (n_pass - 1) * pass_std**2
        + (n_fail - 1) * fail_std**2
    )
    / (n_pass + n_fail - 2)
)
# 여기까지는 pooled std구하기 위해서 추가한 부분





variances = X.var()
X = X.loc[:, variances > 0.01]
print("1차 조정 후 shape:",X.shape)

# 새로운 X_summary DataFrame 생성 준비
X_summary = pd.DataFrame(index=X.columns)

X_summary['Mean'] = X.mean()
X_summary['Std'] = pooled_std
X_summary['CV'] = X.mean()/X.std()
X_summary['PassMean'] = X[y==0].mean()
X_summary['FailMean'] = X[y==1].mean()
X_summary['PFDiff'] = abs(X[y==1].mean()-X[y==0].mean()) / X_summary['Std']

print("\nX의 변수별 평균, 표준편차 등 summary")
print(X_summary.head())

X_top30_summary = X_summary.sort_values('PFDiff', ascending=False).head(30).copy()
print("summary의 PFDiff값 기준으로 내림차순 정렬, 상위 30개")
print(X_top30_summary.head())

X_top30 = X[X_top30_summary.index].copy()

X['Month']=data['Month']

X.head()

X_top30_filtered = X_top30.copy()

for sensor in X_top30.columns :
  Q3 = X_top30[sensor].quantile(0.75)
  Q1 = X_top30[sensor].quantile(0.25)
  IQR = Q3-Q1
  lower = Q1 - 3*IQR
  upper =  Q3 + 3*IQR

  X_top30_filtered[sensor] = X_top30[sensor].where((lower<=X_top30[sensor])&(X_top30[sensor]<=upper), np.nan)

p_values = []

for sensor in X_top30_summary.index:

    pass_values = X.loc[y == 0, sensor]
    fail_values = X.loc[y == 1, sensor]

    t_stat, p_value = ttest_ind(
        pass_values,
        fail_values,
        equal_var=False
    )

    p_values.append(p_value)

X_top30_summary["p_value"] = p_values

# 월별 불량 및 정상 건수 집계하는 monthly_count 만들기
monthly_counts = data.groupby('Month')['Pass/Fail'].value_counts().unstack(fill_value=0)

monthly_counts.rename(columns={1: 'FailureCount', -1: 'NormalCount'}, inplace=True)
monthly_counts['TotalProduction'] = monthly_counts['FailureCount'] + monthly_counts['NormalCount']
monthly_counts['FailureRate'] = (monthly_counts['FailureCount'] / monthly_counts['TotalProduction']) * 100


# DataFrame 인덱스(Month)를 다시 datetime으로 변환하여 그래프에서 정렬하기 쉽게 함
monthly_counts.index = monthly_counts.index.to_timestamp()

# 월별 불량원인 센서 찾기 위한 monthly_summary 만들기
monthly_sensor_summary = []

sensors = X_top30_summary.index.tolist()

for month, month_data in X.groupby('Month'):

    # 해당 월의 Pass / Fail
    month_y = y.loc[month_data.index]

    pass_data = month_data.loc[month_y == 0, sensors]
    fail_data = month_data.loc[month_y == 1, sensors]

    # 센서별 Pass / Fail 평균
    pass_mean = pass_data.mean()
    fail_mean = fail_data.mean()

    # 센서별 표준편차
    std = month_data[sensors].std()

    # PFDiff
    pf_diff = abs(fail_mean - pass_mean) / std

    # 하나의 DataFrame으로 만들기
    temp = pd.DataFrame({
        'Month': month,
        'Sensor': sensors,
        'PassMean': pass_mean,
        'FailMean': fail_mean,
        'PFDiff': pf_diff
    })

    # PFDiff가 큰 순서대로 정렬 후 상위 10개
    temp = temp.sort_values(
        'PFDiff',
        ascending=False
    ).head(10).copy()

    # 등수 부여
    temp['Rank'] = range(1, 11)

    # 원하는 열 순서
    temp = temp[
        ['Month', 'Rank', 'Sensor',
         'PassMean', 'FailMean', 'PFDiff']
    ]

    monthly_sensor_summary.append(temp)


# 모든 월의 결과 합치기
monthly_sensor_summary = pd.concat(
    monthly_sensor_summary,
    ignore_index=True
)


##### 여기부터 streamlit 구현

st.title("XX공장 공정 데이터 기반 불량 분석")
st.caption("캡션 추가")

st.header("<월별 분석>")

col1, col2 = st.columns(2)

# col1 :월별 불량률 꺾은선 그래프

mean_failure_rate = monthly_counts["FailureRate"].mean()


with col1 :
    st.subheader("월별 불량률")
    st.subheader("    ")

    fig, ax = plt.subplots()

    ax.plot(
        monthly_counts.index,
        monthly_counts['FailureRate'],
        marker = '$★$',
        color="#ff8aa1"
    )

    ax.set_xlabel("월")
    ax.set_ylabel("불량률")

    ax.set_title("월별 불량률")

    ax.axhline(
    mean_failure_rate,
    linestyle="--",
    linewidth=1.5
)

    # 평균값 표시
    ax.text(
         1.0,
         mean_failure_rate,
         f"평균 {mean_failure_rate/100:.1%}",
         transform=ax.get_yaxis_transform(),
         ha="left",
         va="bottom",
         fontsize=12
    )
    ax.grid(True)

    st.pyplot(fig)


# col2 : 월을 선택한 뒤 월별 PFDifference가 큰 센서 5개를 보여줌
with col2 :
    st.subheader("불량 의심 센서 Top5")
    monthly_counts.index = monthly_counts.index.to_period('M')
    options = monthly_counts.index.astype(str).tolist()
    monthly_counts.index = monthly_counts.index.to_timestamp()

    selected_month = st.selectbox(
        "월별 보기",
        options
    )

    if selected_month == '월별 보기':
        pass
    else : 
        selected_month = pd.Period(selected_month, freq="M")

        failure_count = monthly_counts.loc[
            selected_month.to_timestamp(),
            "FailureCount"
        ]
                
        if failure_count <=5 :
            st.warning(
                f"{selected_month}월은 불량 데이터가 {failure_count}건으로 적어 "
                "불량 의심 센서 분석 결과를 제공하지 않습니다."
            )
        else :
            result = monthly_sensor_summary[
                monthly_sensor_summary["Month"] == selected_month
            ].head(5)

            fig, ax = plt.subplots()

            ax.barh(
                result["Rank"],
                result["PFDiff"],
                color="#f7b0ab"
            )

            ax.set_yticks(result["Rank"])
            ax.set_yticklabels(
                [f"센서 {sensor}" for sensor in result["Sensor"]]
            )

            ax.invert_yaxis()

            ax.set_xlabel("PFDifference")
            ax.set_ylabel("Top 5")
            ax.set_title(f"{selected_month} 불량 의심 센서")

            ax.grid(axis="x")

            st.pyplot(fig)


#  아래쪽~~

st.subheader("전기간 불량 의심 센서 Top 6")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

top3_sensors = X_top30_summary.head(3).index

for ax, sensor in zip(axes, top3_sensors):

    pass_data = X_top30_filtered.loc[y == 0, sensor].dropna()
    fail_data = X_top30_filtered.loc[y == 1, sensor].dropna()

    plot_data = pd.DataFrame({
        "Value": pd.concat([pass_data, fail_data]),
        "Group": (
            ["Pass"] * len(pass_data)
            + ["Fail"] * len(fail_data)
        )
    })

    sns.violinplot(
        data=plot_data,
        x="Group",
        y="Value",
        inner=None,
        color="#f5ea78",
        ax=ax
    )

    for i, group in enumerate(["Pass", "Fail"]):

        group_data = plot_data.loc[
            plot_data["Group"] == group, "Value"
        ]

        Q1 = group_data.quantile(0.25)
        Q2 = group_data.quantile(0.50)
        Q3 = group_data.quantile(0.75)

        ax.hlines(Q1, i - 0.3, i + 0.3, color="#ff9d80", linewidth=1.5)
        ax.hlines(Q2, i - 0.5, i + 0.5, color="#ff9d80",linewidth=2.5)
        ax.hlines(Q3, i - 0.3, i + 0.3, color="#ff9d80", linewidth=1.5)

    pfdiff = X_top30_summary.loc[sensor, "PFDiff"]
    
    ax.text(
        0.97, 0.95,
        f"PFDiff = {pfdiff:.3f}\np < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12
    )
    ax.set_title(f"Sensor {sensor}", fontsize=15)
    ax.set_xlabel("")
    ax.set_ylabel("센서값", fontsize=16)
    ax.tick_params(axis='x', labelsize=20, color='#db1616')

plt.tight_layout()

st.pyplot(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

top6_sensors = X_top30_summary.iloc[3:6].index

for ax, sensor in zip(axes, top6_sensors):

    pass_data = X_top30_filtered.loc[y == 0, sensor].dropna()
    fail_data = X_top30_filtered.loc[y == 1, sensor].dropna()

    plot_data = pd.DataFrame({
        "Value": pd.concat([pass_data, fail_data]),
        "Group": (
            ["Pass"] * len(pass_data)
            + ["Fail"] * len(fail_data)
        )
    })

    sns.violinplot(
        data=plot_data,
        x="Group",
        y="Value",
        inner=None,
        color="#bff5f3",
        ax=ax
    )

    for i, group in enumerate(["Pass", "Fail"]):

        group_data = plot_data.loc[
            plot_data["Group"] == group, "Value"
        ]

        Q1 = group_data.quantile(0.25)
        Q2 = group_data.quantile(0.50)
        Q3 = group_data.quantile(0.75)

        ax.hlines(Q1, i - 0.3, i + 0.3, color="#383b37", linewidth=1.5)
        ax.hlines(Q2, i - 0.5, i + 0.5, color="#383b37",linewidth=2.5)
        ax.hlines(Q3, i - 0.3, i + 0.3, color="#383b37", linewidth=1.5)

    pfdiff = X_top30_summary.loc[sensor, "PFDiff"]
        
    ax.text(
            0.97, 0.95,
            f"PFDiff = {pfdiff:.3f}\np < 0.05",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12
        )
    ax.set_title(f"Sensor {sensor}", fontsize=15)
    ax.set_xlabel("")
    ax.set_ylabel("센서값", fontsize=16)
    ax.tick_params(axis='x', labelsize=20, color='#db1616')

plt.tight_layout()

st.pyplot(fig)

st.subheader(" ")
st.dataframe(X_top30_summary.head(6))