import torch
import torch.nn as nn
import random

from model.kobert import KoBERTforSequenceClassfication, kobert_input
from kobert_transformers import get_tokenizer
#데이터를 불러와서 감정과 답변을 사전형태로 출력하는 함수
#data/chatbot_wellness_category.txt파일은 아마 감정    답변 형태의 파일일 겁니다
def load_answer():
  root_path = '..' #'.'는 현재 이곳의 경로 '..'는 부모폴더의 경로를 뜻함(https://hongl.tistory.com/289참고) 
  answer_path = f"{root_path}/data/chatbot_wellness_category.txt" #이 경로에서 chatbot_wellness_category.txt파일을 불러옴

  a_f = open(answer_path,'r') #데이터를 읽는다
  answer_lines = a_f.readlines() #answer_lines리스트에 한줄한줄이 원소로 들어간다(https://wikidocs.net/82참고)
  answer = {}

  #{답변:감정}형태의 answer사전을 형성 - data[1][:-1]에 대한 자세한 설명은 4.10폴더를 참고하세요
  for line_num, line_data in enumerate(answer_lines):#enumerate함수는 (인덱스,내용)의 튜플 형태로 만들어준다(https://www.daleseo.com/python-enumerate/참고) line_num에 인덱스, line_data에 내용이 들어가는듯
    
    data = line_data.split('    ')#문자열을 뛰어쓰기 네번을 기준으로 짤라서 data리스트를 형성한다
    #현재 data는 1차원 배열이지만 문자열은 2차원 배열처럼 볼수 있다고 합니다 
    answer[data[1][:-1]] =data[0] 
    #answer={답변1:감정1,답변2:감정2,답변3:감정3,...}의 형태로 답변이 저장돼있다.

  return  answer

if __name__ == "__main__": 
  root_path='..'
  checkpoint_path =f"{root_path}/checkpoint"
  save_ckpt_path = f"{checkpoint_path}/kobert-chatbot-wellness.pth"#.pth 파일은 학습한 모델의 정보가 들어있는 파일정도로 이해(https://qlsenddl-lab.tistory.com/41)

  #답변 불러오기
  answer = load_answer()

  #모델을 학습할때 gpu사용이 가능하면 쓰고 아니면 cpu를 쓴다
  ctx = "cuda" if torch.cuda.is_available() else "cpu" 
  device = torch.device(ctx)

  # 저장한 Checkpoint 불러오기
  checkpoint = torch.load(save_ckpt_path, map_location=device)

  model = KoBERTforSequenceClassfication(num_labels=9322)
  model.load_state_dict(checkpoint['model_state_dict']) #save_ckpt_path파일 안에 매개변수model_state_dict 정보(아마도 가중치나 편향값 같은 것)를 불러온다
  model.eval() #파일 입출력할 때 마지막에 close함수로 닫아주는 것처럼 .load_state_dict함수를 쓴 후 해줘야하는 것으로 무시해도 된다.
  #load_state_dict,state_dict, model_state_dict에관한 내용 모두 https://tutorials.pytorch.kr/beginner/saving_loading_models.html 참고)
  tokenizer = get_tokenizer() 
  #사용자로 부터 문장을 입력받고 모델을 통해 답변을 추출한다
  while 1:
    sent = input('\nQuestion: ') # '요즘 기분이 우울한 느낌이에요'
    data = kobert_input(tokenizer,sent, device,512) #모델의 입력으로 들어가기위한 처리(토크나이즈, 패딩, attetion mask지정 등등)
    # print(data)

    output = model(**data) # **에 관한 내용(https://dojang.io/mod/page/view.php?id=2347)

    logit = output
    softmax_logit = nn.Softmax(logit).dim #Softmax함수는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록 logit값을 [0, 1] 범위로 비례하여 조정한다
    softmax_logit = softmax_logit[0].squeeze() #.squeeze()함수는 그냥 해당값을 불러오는 함수정도로 이해하면 됩니다

    max_index = torch.argmax(softmax_logit).item() # torch.argmax(텐서)는 텐서에서 최댓값이 들어있는 곳의 인덱스를 출력한다 https://velog.io/@jarvis_geun/torch.argmax-torch.max 참고
    max_index_value = softmax_logit[torch.argmax(softmax_logit)].item() #.item()은 사전형식의 변수에 대해 키와 내용 모두를 출력한다. https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sw4r&logNo=221504133335 참고

    print(f'Answer: {answer[str(max_index)]}, index: {max_index}, value: {max_index_value}')
    print('-'*50)
  # print('argmin:',softmax_logit[torch.argmin(softmax_logit)])
