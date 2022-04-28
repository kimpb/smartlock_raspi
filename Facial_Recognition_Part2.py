import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'faces/'
#사진 경로 지정
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
#faces 폴더에 있는 파일 리스트 얻기
Training_Data, Labels = [], []
#데이터와 매칭될 라벨 변수
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    #파일 개수만큼 루프
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #이미지 불러오기
    if images is None:
        continue
    #이미지 파일이 없으면 무시
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    #Training_Data 리스트에 이미지를 바이트 배열로 추가
    Labels.append(i)
    #Labels 리스트엔 카운트 번호 추가

Labels = np.asarray(Labels, dtype=np.int32)
#Labels를 32비트 정수로 변환

model = cv2.face.LBPHFaceRecognizer_create()
#모델 생성
model.train(np.asarray(Training_Data), np.asarray(Labels))
#학습 시작
print("Model Training Complete!!!!!")
