# 서비스를위해서는 사용자 input이 필요함
# 환자의 수치를 이용해 얼마 나올지 예측하고 싶음. 
# 학습시킨 후 예측 예제: [[16.34, 87.21]] 
# input 은 기본 형식이 string으로 들어옴. 고로 나중에 int or float로 바꿔야함._머신러닝때 학습했던 형식으로 
## ⇒ input() return str() → float()
texture_mean= float(input('texture_mean : '))
perimeter_mean= float(input('perimete_mean : '))

# pickle로 만든 파일을 가져와야함. 

import pickle
## study_pythons에 exist 확인하는거 다시 보기. 
with open('datasets/BreastCancerWisconsin_Regression.pkl','rb') as regression_file:
    ## pickle 인스턴스화
    loaded_model = pickle.load(regression_file)
    input_labels= [[texture_mean, perimeter_mean]] ## 머신러닝할때 학습시킨 설명변수 형식 그대로. 
    result_predict = loaded_model.predict(input_labels)
    print('Predicted radius_mean : {}'.format(result_predict))
    pass
## 예측 결과값: Predicted radius_mean : [13.45096511]
## 주피터랩에서 한 예측 값이 동일함. 
