import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import bluetooth
import signal
import sys
from select import *

HOST = ""
PORT = bluetooth.PORT_ANY
UUID = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

GPIO.setmode(GPIO.BCM)
trig = 14 # trig = 14 초음파 신호 전송 핀번호 지정
echo = 15 # echo = 15 초음파 신호 수신 핀번호 지정
GPIO.setup(2, GPIO.OUT) # 릴레이 모듈 GPIO output 설정
GPIO.setup(trig, GPIO.OUT) # trig 초음파 신호 전송핀 번호 output 설정
GPIO.setup(echo, GPIO.IN) # echo 초음파 신호 수신핀 번호 input 설정
GPIO.setwarnings(False)
GPIO.output(2, False)

from os import listdir
from os.path import isdir, isfile, join

def signal_handler(sig, frame):
    try:
        connected_socket.close()

    except:
        pass

    server_socket.close()
    sys.exit()

cnt = 0
status = 0
# 얼굴 인식용 haar/cascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    

# 사용자 얼굴 학습
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로 만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    #학습 모델 리턴
    return model

# 여러 사용자 학습
def trains():
    #faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
    #학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('model :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('model2 :' + model)
        models[model] = result

    # 학습된 모델 딕셔너리 리턴
    return models    

#얼굴 검출
def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is():
            return img,[]
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

# 인식 시작
def run(models):    
    count = 0
    unknowncount = 0
    #카메라 열기 
    cap = cv2.VideoCapture(-1)
    while True:
        #카메라로 부터 사진 한장 읽기 
        ret, frame = cap.read()
        # 얼굴 검출 시도 
        image, face = face_detector(frame)
        try:            
            min_score = 999       #가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""   #가장 높은 점수로 예측된 사람의 이름
            
            #검출된 사진을 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            

            #위에서 학습한 모델로 예측시도
            for key, model in models.items():
                result = model.predict(face)                
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                    
            #min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.         
            if min_score < 500:
                
                confidence = int(100*(1-(min_score)/300))
                # 유사도 화면에 표시 
                display_string = str(confidence)+'% Confidence it is ' + min_score_name
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            #75 보다 크면 동일 인물로 간주해 UnLocked! 
            if confidence > 75:
                cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                print("Unlocked")
                count = 0
                GPIO.output(2, True)
                time.sleep(5)
                GPIO.output(2, False)
                break
                
            else:
            #75 이하면 타인.. Locked!!! 
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
                print("Unknown")    
                count = 0
                unknowncount += 1
                GPIO.output(2, False)
                time.sleep(0.1)
                if unknowncount > 30:
                	print("CARE! UNKNOWN USER!")
                	status = 0
                	break
        except:
            #얼굴 검출 안됨 
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            GPIO.output(2, False)
            print("Face not found")
            time.sleep(0.1)
            count += 1
            if count > 100:
              count = 0
              status = 0
              break
        if cv2.waitKey(1)==13:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 학습 시작
    models = trains()
    # 실행



while True:
        
        GPIO.output(trig,False)
        time.sleep(0.5)
        GPIO.output(trig,True)
        time.sleep(0.000001)
        GPIO.output(trig,False)
        while GPIO.input(echo) == 0:
            start = time.time()
        while GPIO.input(echo) == 1:
            end = time.time()
        distance = (end - start) * 17000
        distance = round(distance,2)
        print ('Dist =', distance)

        if distance < 100:
            cnt += 1
            if cnt%5 == 0:
                run(models)
            pass
        else:
            cnt = 0
            pass
        


