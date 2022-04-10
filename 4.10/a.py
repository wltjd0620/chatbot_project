
#제가 예측하기로 chatbot_wellness_category.txt파일은 제가 폴더에 넣은 파일과 같은 파일입니다
def load_answer():
  root_path = '.' #'.'는 현재 이곳의 경로 '..'는 부모폴더의 경로를 뜻함(https://hongl.tistory.com/289참고) 
  answer_path = f"{root_path}/chatbot_wellness_category.txt" #이 경로에서 chatbot_wellness_category.txt파일을 불러옴

  a_f = open(answer_path,'r',encoding="UTF-8") #데이터를 읽는다
  answer_lines = a_f.readlines() #answer_lines리스트에 한줄한줄이 원소로 들어간다(https://wikidocs.net/82참고)
  answer = {}

  #데이터의 감정을 담은 answer사전을 형성
  for line_num, line_data in enumerate(answer_lines):#enumerate함수는 (인덱스,내용)의 튜플 형태로 만들어준다(https://www.daleseo.com/python-enumerate/참고) line_num에 인덱스, line_data에 내용이 들어가는듯
    print(line_data)
    data = line_data.split('    ')#문자열을 뛰어쓰기 네번을 기준으로 짤라서 data리스트를 형성한다
    #현재 data는 1차원 배열이지만 문자열은 2차원 배열처럼 볼수 있다고 합니다 제 파일 기준 for문의 첫번째 반복에서 data[0]=분노, data[1]=너무 화나/n 입니다. data[1][:-1]=너무 화나입니다
    answer[data[1]] =data[0] 
    #즉 answer=[답변1:감정1,답변2:감정2,답변3:감정3,...]의 형태로 답변이 저장돼있다.
    print(answer)
  return  answer

load_answer()