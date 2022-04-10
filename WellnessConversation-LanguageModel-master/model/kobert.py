import torch
import torch.nn as nn
# https://pytorch.org/docs/stable/nn.html
# torch.nn 내부 함수, 변수 목록
# CrossEntropyLoss = 이 기준은 입력과 목표 사이의 교차 엔트로피 손실을 계산합니다.
# MESLoss = https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
# CLASStorch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
# 
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from kobert_transformers import get_kobert_model, get_distilkobert_model
from model.configuration import get_kobert_config

"""
kobert config
predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
"""
# KoBERT 모델 을 이용해 대화 및 답변 텍스트 생성
class KoBERTforSequenceClassfication(BertPreTrainedModel):
  def __init__(self,
                num_labels = 359, # 분류할 라벨 갯수를 설정
                hidden_size = 768, # hidden_size
                hidden_dropout_prob = 0.1, #dropout_prop
               ):
    # super() = 자식 클래스에서 부모 클래스의 내용을 사용하고 싶을 경우 사용 - 상속 내용
    # https://jimmy-ai.tistory.com/79
    # super().__init__ 
    # = 다른 클래스의 속성 및 메소드를 자동으로 불러와 해당 클래스에서도 사용이 가능하도록 해줍니다.
    # 여기서는 configuration.py의 get_kobert_config() 함수의 메소드를 불러온다.
    super().__init__(get_kobert_config())
    
    self.num_labels = num_labels
    self.kobert = get_kobert_model()
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.classifier = nn.Linear(hidden_size, num_labels)

    self.init_weights()
  
  # forward 함수( )
  def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          labels=None,
  ):
    outputs = self.kobert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

    if labels is not None:
      if self.num_labels == 1:
        #  회귀분석 시작
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), logits, (hidden_states), (attentions)
# input 값 정리
def kobert_input(tokenizer, str, device = None, max_seq_len = 512):
  index_of_words = tokenizer.encode(str)
  token_type_ids = [0] * len(index_of_words)
  attention_mask = [1] * len(index_of_words)

  # Padding Length
  padding_length = max_seq_len - len(index_of_words)

  # Zero Padding
  index_of_words += [0] * padding_length
  token_type_ids += [0] * padding_length
  attention_mask += [0] * padding_length

  data = {
    'input_ids': torch.tensor([index_of_words]).to(device),
    'token_type_ids': torch.tensor([token_type_ids]).to(device),
    'attention_mask': torch.tensor([attention_mask]).to(device),
  }
  return data
