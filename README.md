# -

# 반도체 공정 불량 분석 프로젝트

UCI SECOM 데이터를 활용하여 반도체 공정의 정상(Pass)과 불량(Fail) 데이터를 분석하고,
센서별 특성과 불량 의심 센서를 탐색한 프로젝트입니다.

## 프로젝트 목적

- 반도체 공정 센서 데이터의 전처리 및 탐색적 데이터 분석
- Pass / Fail 그룹 간 센서값 차이 분석
- 월별 불량률 및 불량 의심 센서 분석
- 센서별 효과크기(Cohen's d 기반 PFDiff) 분석
- 머신러닝을 활용한 불량 예측 시도
- Streamlit을 활용한 분석 결과 시각화

## 주요 분석

### 1. 월별 불량률
월별 전체 생산량과 불량 생산량을 비교하여 불량률 변화를 확인합니다.

### 2. 월별 불량 의심 센서
각 월의 Pass / Fail 센서값 차이를 PFDiff 기준으로 비교하여
불량과 관련성이 높은 센서를 확인합니다.

### 3. 센서별 분석
전체 기간을 기준으로 주요 센서의 Pass / Fail 분포를
Violin Plot 등을 활용하여 비교합니다.

### 4. 머신러닝
미구현

## 사용 기술

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SMOTE
- Streamlit

## 데이터

UCI SECOM Dataset

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
