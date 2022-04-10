import logging # 어떤 소프트웨어가 실행될 때 발생하는 이벤트를 추적하는 수단.
# logging 관련 공부자료
# https://docs.python.org/ko/3/howto/logging.html#logging-basic-tutorial

#transformers = 
from transformers.configuration_utils import PretrainedConfig
from transformers import BertModel, BertConfig, GPT2Config


# 
logger = logging.getLogger(__name__)

#이 아래는 각 모델에 대한 기본값들을 설정해주는 부분이다.
# 각 라이브러리 모델의 변수값 설정
#KoBERT
kobert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 8002
}
#KoGPT2
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "activation_function": "gelu"
}
# kobert_config 값을 상속한다. ( 정확하게 무슨 역할인지 모르겠음 )
def get_kobert_config():
    return BertConfig.from_dict(kobert_config)

def get_kogpt2_config():
    return GPT2Config.from_dict(kogpt2_config)