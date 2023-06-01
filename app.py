from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np


# st.title("자동 채점 모델 기반 자동 피드백")
# st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")

st.subheader("문항 1-8")
st.markdown("저장 매체의 용량을 나타내는 단위로 $B, KB, MB$ 등이 있고, $1KB=2^{10}B, 1MB=2^{10}KB$이다. 찬혁이가 컴퓨터로 용량이 $36MB$인 자료를 내려받으려고 한다. 이 컴퓨터에서 1초당 내려받는 자료의 용량이 $9×{2^{20}}B$일 때, 찬혁이가 자료를 모두 내려받는 데 몇 초가 걸리는지 구하시오.")
st.write("풀이과정과 답을 아래 답안 란에 자세하게 작성해 주세요. 여러분이 알고 있는 개념과 오개념을 파악하는 데 도움을 줄 것입니다.")
response = st.text_input('답안 :', key='answer_input_1_8')

#모델의 이름 정하기
model_name_1_8 = "1-8_rnn_sp_140" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 140 #vocab size
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d_1_8 = 5 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)

model_1_8 = RNNModel(output_d_1_8, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
# model = ATTModel(output_d, c) #ATTModel 쓰는경우

model_1_8.load_state_dict(torch.load("./save/"+model_name_1_8+".pt"))

#자신에게 맞는 모델로 부르기
tokenizer_1_8 = AutoTokenizer.from_pretrained("./save/"+ model_name_1_8) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name_1_8+"-vocab.txt") #bw tokenizer 쓰는경우

enc = tokenizer_1_8(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model_1_8(pad_ten)
label_1_8 = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기', key='button_1_8_1'):
    """
    output차원에 맞추어 피드백 넣기
    """
    st.write(response)
    #if len(label_1_8)>= 5:
        if label_1_8[0] == 1 and label_1_8[1] == 1 and label_1_8[2] == 1 and label_1_8[3] == 0 and label_1_8[4] == 0:
         st.success('거듭제곱의 곱셈, 거듭제곱의 나눗셈, 단위의 변환을 잘하는구나!', icon="✅")
        elif label_1_8[0] == 1 and label_1_8[1] == 0 and label_1_8[2] == 0 and label_1_8[3] == 1 and label_1_8[4] == 1:
         st.success('거듭제곱의 나눗셈, 수의 나눗셈, 단위의 변환을 잘하는구나!', icon="✅")
    else :
        st.info('거듭제곱의 곱셈, 거듭제곱의 나눗셈, 수의 나눗셈, 단위의 변환을 잘 생각해보자!', icon="ℹ️")
else : 
    st.button('피드백 받기 버튼을 눌러보세요!')
