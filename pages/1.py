import streamlit as st
import torch
import numpy as np


st.title("자동 채점 및 피드백 모델")
st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")

st.subheader("1. 기획 의도")

st.write("과정 중심 평가에 이용하는 서술형 평가 문제의 답안을 자동 채점 모델을 통하여 신속하게 채점하고 각 학생들에게 부족한 인지요소 및 오개념을 파악한 후, 결과를 토대로 학생들에게 피드백을 줄 수 있는 것이 이 프로젝트의 목표이자 의도임.")
st.write("이를 통해 첫 시간의 화두였던 "성취 지식 및 교과 역량에 대한 맞춤형 평가" 를 기대할 수 있음")
