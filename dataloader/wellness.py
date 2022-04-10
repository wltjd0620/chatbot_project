import torch
import torch.nn as nn
from torch.utils.data import Dataset # 데이터로더

from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

#데이터셋, 데이터로더의 개념에 관한 내용은 https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html 참고
#이 파일에서 WellnessAutoRegressiveDataset와 WellnessTextClassificationDataset 모두 사용자 정의 데이터셋을 만드는 클래스인대 토크나이저에 따라 두 개의 클래스로 나눠놓은 것 같아요
# 코드의 내용은 토크나이저에 따라 토크나이징하는 방식이 다르기 때문에 변수가 다르고 내용이 다른 것뿐 기능적으로 동일한 클래스라고 추측됩니다.

# 사용자 정의 데이터셋을 만든다
class WellnessAutoRegressiveDataset(Dataset): #Dataset이라는 데이터로더를 쓰겠다
  """Wellness Auto Regressive Dataset"""
  #문장과 그에 대한 정답(ex감정)이 든 데이터를 불러와 토큰화하고 패딩처리하여 self.data(리스트)에 저장한다.
  def __init__(self,
               file_path = "../data/wellness_dialog_for_autoregressive.txt",
               n_ctx = 1024
               ):
    self.file_path = file_path
    self.data =[]
    self.tokenizer = get_kogpt2_tokenizer() #이 토크나이저에 대한 내용을 알아야 bos_token_id같은 변수가 무엇인지 알 수 있을 것 같은대 찾지 못했습니다.


    bos_token_id = [self.tokenizer.bos_token_id]
    eos_token_id = [self.tokenizer.eos_token_id]
    pad_token_id = [self.tokenizer.pad_token_id]

    file = open(self.file_path, 'r', encoding='utf-8')
    
    #모든 데이터들을 모델입력으로 들어가기 위한 처리를 한후에 self.data리스트에 넣는다
    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("    ")
      index_of_words = bos_token_id +self.tokenizer.encode(datas[0]) + eos_token_id + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
      pad_token_len = n_ctx - len(index_of_words)

      index_of_words += pad_token_id * pad_token_len

      self.data.append(index_of_words) #self.data리스트에 넣는다

    file.close()

  #데이터셋의 샘플 개수를 반환합니다.
  def __len__(self):
    return len(self.data)

  #주어진 인덱스 index 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다.
  def __getitem__(self,index):
    item = self.data[index]
    return item

class WellnessTextClassificationDataset(Dataset):
  """Wellness Text Classification Dataset"""
  #문장과 그에 대한 정답(ex감정)이 든 데이터를 불러와 토큰화하고 패딩처리하여 self.data(리스트)에 저장한다.
  def __init__(self,
               file_path = "../data/wellness_dialog_for_text_classification.txt",
               num_label = 359,
               device = 'cpu',
               max_seq_len = 512, # KoBERT max_length
               tokenizer = None
               ):
    self.file_path = file_path
    self.device = device
    self.data =[]
    self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer() #클래스 사용시 토크나이저를 명시하지 않으면 kobert_transformers의 토크나이저를 쓴다.


    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("    ")
      index_of_words = self.tokenizer.encode(datas[0])
      token_type_ids = [0] * len(index_of_words) #토큰이 존재하는 부분은 0으로 채운다
      attention_mask = [1] * len(index_of_words) #토큰이 존재하는 부분은 1로 채운다

      # Padding Length
      padding_length = max_seq_len - len(index_of_words) #모델의 입력문장길이가 max_seq_len이니까 토큰이 존재하는 않는 부분이 패딩처리할 길이가 된다.

      # Zero Padding (max_seq_len길이에 맞춰 모두 패딩처리한다)
      index_of_words += [0] * padding_length
      token_type_ids += [0] * padding_length
      attention_mask += [0] * padding_length

      # Label
      label = int(datas[1][:-1])

      #위에서 처리한 변수들은 cpu 또는 gpu에 정보가 저장돼있는대 .to함수를 이용해 현재 정해진 디바이스(cpu)로 정보를 옮긴다
      # - (제가 알기로 tensor로 생성한 변수는 cpu에 자동으로 저장되는대 여기서는 device가 cpu가 아닐 경우를 위해 코드를 작성해놓은 것 같아요)
      data = {
              'input_ids': torch.tensor(index_of_words).to(self.device),
              'token_type_ids': torch.tensor(token_type_ids).to(self.device),
              'attention_mask': torch.tensor(attention_mask).to(self.device),
              'labels': torch.tensor(label).to(self.device)
             }

      self.data.append(data)

    file.close()

  #데이터셋의 샘플 개수를 반환합니다.
  def __len__(self):
    return len(self.data)
  #주어진 인덱스 index 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다.
  def __getitem__(self,index):
    item = self.data[index]
    return item

if __name__ == "__main__":
  dataset = WellnessAutoRegressiveDataset()
  dataset2 = WellnessTextClassificationDataset()
  print(dataset)
  print(dataset2)