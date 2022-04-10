import torch
import torch.nn as nn
import random
from transformers import (
  ElectraConfig,
  ElectraTokenizer
)
from model.koelectra import koElectraForSequenceClassification,koelectra_input
from model.kobert import KoBERTforSequenceClassfication, kobert_input
from kobert_transformers import get_tokenizer

#DialogKoBERT.predict로 이 클래스를 사용하게 될거라고 추측하는데 predict함수에 대한 내용이 곧 이 클래스의 전부라고 보면 될 것 같습니다.
class DialogKoBERT:
    #생성자로서 어떤 디바이스를 쓸건지 어떤 모델을 쓸건지 어떤 토크나이저를 쓸건지 등 잡다한 정보가 들어있다.
    def __init__(self):
        self.root_path='..'
        self.checkpoint_path =f"{self.root_path}/checkpoint"
        self.save_ckpt_path = f"{self.checkpoint_path}/kobert-wellnesee-text-classification.pth" #.pth 파일은 학습한 모델의 정보가 들어있는 파일정도로 이해(https://qlsenddl-lab.tistory.com/41)
        #답변과 카테고리 불러오기
        self.category, self.answer = load_wellness_answer() #load_wellness_answer함수는 이 파일 아랫부분 보면 정의돼있는대 답변과 감정 데이터를 사전형식으로 출력하는 함수(출력값 두 개)입니다.

        #cpu ,gpu 중 쓸 디바이스를 설정한다
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        # 저장한 Checkpoint 불러오기
        checkpoint = torch.load(self.save_ckpt_path, map_location=self.device)

        self.model = KoBERTforSequenceClassfication()
        self.model.load_state_dict(checkpoint['model_state_dict']) #save_ckpt_path파일 안에 매개변수model_state_dict 정보(아마도 가중치나 편향값 같은 것)를 불러온다

        self.model.eval() #파일 입출력할 때 마지막에 close함수로 닫아주는 것처럼 .load_state_dict함수를 쓴 후 해줘야하는 것으로 무시해도 된다.

        self.tokenizer = get_tokenizer() #kobert_transformers의 토크나이저를 쓰겠다

    #문장이 입력으로 들어오면 모델을 거쳐서 정답(감정)을 추측하고 그 감정에 해당하는 답변을 출력하는 함수이다.
    #어떤 문장이 입력으로 들어가면 출력은 각 감정(예를 들어 10가지라고 하면)에 대한 확률값 10개가 나온다 그 중 가장 큰값으로 정답을 내리게 된다.
    def predict(self, sentence):
        data = kobert_input(self.tokenizer, sentence, self.device, 512) #kobert_input함수는 모델의 입력으로 들어가기위한 처리를 한다(토크나이즈, 패딩, attetion mask지정 등등)
        output = self.model(**data) #모델의 출력값이 어떤 형태인지 몰라서 아직은 정확한 코드분석이 어렵습니다.
        logit = output
        softmax_logit = nn.Softmax(logit).dim #Softmax함수는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록 logit값을 [0, 1] 범위로 비례하여 조정한다
        softmax_logit = softmax_logit[0].squeeze() #softmax_logit에 그 확률값을 저장한다.
        #.squeeze()는 그냥 해당 부분을 불러오는 함수 정도로 이해하면 됩니다. (https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html 의 DataLoader를 통해 순회하기 참고)

        max_index = torch.argmax(softmax_logit).item() # torch.argmax(텐서)는 텐서에서 최댓값이 들어있는 곳의 인덱스를 출력한다 https://velog.io/@jarvis_geun/torch.argmax-torch.max 참고
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item() #.item()은 사전형식의 변수에 대해 키와 내용 모두를 출력한다. https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sw4r&logNo=221504133335 참고

        answer_list = self.answer[self.category[str(max_index)]] #answer가 사전형식이니까 self.answer[?]에서 ?는 키(감정)를 의미하고 self.answer[?]는 그 키에 해당하는 답변이다.
        answer_len= len(answer_list)-1 #답변 데이터의 총 개수를 의미하며, 데이터의 첫번째 값이 쓸모없는 값이 들어있을 거라고 추측된다.
        answer_index = random.randint(0,answer_len) #하나의 감정 분류에 대한 답변은 여러개 일 것이므로 그 중 랜덤하게 답변을 추출하기 위해 랜덤한 인덱스를 설정해준다

        return answer_list[answer_index]
        
#이 클래스의 내용은 위 내용과 동일 하므로 주석처리하지 않겠습니다.
class DialogElectra:
    def __init__(self):
        self.root_path = '..'
        self.checkpoint_path = f"{self.root_path}/checkpoint"
        self.save_ckpt_path = f"{self.checkpoint_path}/koelectra-wellnesee-text-classification.pth"
        model_name_or_path = "monologg/koelectra-base-discriminator"

        # 답변과 카테고리 불러오기
        self.category, self.answer = load_wellness_answer() 

        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        # 저장한 Checkpoint 불러오기
        checkpoint = torch.load(self.save_ckpt_path, map_location=self.device)

        # Electra Tokenizer
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path) #토크나이저를 설정한다 .from_pretrained는 그냥 ()안 경로의 것을 불러온다 정도의 의미니까 무시해도 됩니다.

        electra_config = ElectraConfig.from_pretrained(model_name_or_path)
        self.model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                                   config=electra_config,
                                                                   num_labels=359)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence):
        data = koelectra_input(self.tokenizer, sentence, self.device, 512)
        # print(data)

        output = self.model(**data)

        logit = output
        softmax_logit = nn.Softmax(logit).dim
        softmax_logit = softmax_logit[0].squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        answer_list = self.answer[self.category[str(max_index)]] 
        answer_len = len(answer_list) - 1
        answer_index = random.randint(0, answer_len)
        return answer_list[answer_index]

# 데이터에서 답변과 감정 정보를 각각 answer, category(사전형식)에 넣고 그 두 변수가 출력된다.
def load_wellness_answer():
  root_path = '..'
  category_path = f"{root_path}/data/wellness_dialog_category.txt"
  answer_path = f"{root_path}/data/wellness_dialog_answer.txt"

  c_f = open(category_path,'r')
  a_f = open(answer_path,'r')

  category_lines = c_f.readlines() #readlines()는 모든 정보 읽기 readline()는 한줄만 읽기
  answer_lines = a_f.readlines()

  category = {}
  answer = {}
  #data[1][:-1]에 대한 자세한 설명은 4.10폴더를 참고하세요
  for line_num, line_data in enumerate(category_lines): #enumerate함수는 (인덱스,내용)의 튜플 형태로 만들어준다(https://www.daleseo.com/python-enumerate/참고)
    data = line_data.split('    ') #문자열을 뛰어쓰기 네번을 기준으로 짤라서 data리스트를 형성한다
    category[data[1][:-1]]=data[0]

  for line_num, line_data in enumerate(answer_lines):
    data = line_data.split('    ')
    keys = answer.keys()
    if(data[0] in keys):
      answer[data[0]] += [data[1][:-1]]
    else:
      answer[data[0]] =[data[1][:-1]]

  return category, answer