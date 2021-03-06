import bluetooth
import signal
import sys
import RPi.GPIO as GPIO
import time
import socket
from _thread import *
from encodings import utf_8
import base64
from Crypto import Random
from Crypto.Cipher import AES

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)
GPIO.output(2, False)

svHOST = '13.209.98.41'
svPORT = 4000
btHOST = ""  # '블루투스 컨트롤러 맥 주소'를 직접 입력해도 됨
btPORT = bluetooth.PORT_ANY
UUID = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

global iv
iv = '0123456789012345' # 16bit

BS = AES.block_size
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
unpad = lambda s : s[0:-ord(s[-1])]
global otp_key
otp_key = 'QWERQWERQWERQWERQWERQWERQWERQWER'
key = otp_key.encode('utf-8')

def svrecv_data(svclient_socket):
    while True:
        global decryptotp
        svrecvKEY = svclient_socket.recv(1024)
        svrecvKEY = svrecvKEY.decode()
        print("server OTP : ",svrecvKEY)
        cipher = AES.new(otp_key.encode("utf8"), AES.MODE_CBC, IV=iv.encdoe("utf8"))
        decryptotp = cipher.decrypt(base64.b64decode(svrecvKEY))
        decryptotp = unpad(decryptotp.decode("utf8"))
        print(decryptotp)
        global stop_threads
        stop_threads = True
        if stop_threads:
            svclient_socket.close()
            break

def btrecv_data(btclient_socket):
    while True:
        global btrecvKEY
        btrecvKEY = btclient_socket.recv(1024)
        btrecvKEY = btrecvKEY.decode()
        print("bluetooth client : ",btrecvKEY)


def signal_handler(sig, frame):
    try:
        btconnected_socket.close()

    except:
        pass

    btserver_socket.close()
    sys.exit()

while True:
    signal.signal(signal.SIGINT, signal_handler)

    stop_threads = False

# 블루투스 서버 소켓 생성
    btserver_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    btserver_socket.bind((btHOST, btPORT))
    btserver_socket.listen(1)

    btport = btserver_socket.getsockname()[1]
    print("bluetooth socket create, port is :", btport)

# 블루투스 서비스 advertise
    bluetooth.advertise_service(
        btserver_socket,
        name="server",
        service_id=UUID,
        service_classes=[UUID, bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE],
    )

# 클라이언트 접속 대기
    btconnected_socket, client_address = btserver_socket.accept()
    btrecvKEY = None
    print(btrecvKEY)
    print("bluetooth connected")

    try:
        while True:
            btrecvKEY = btconnected_socket.recv(1024)
            btrecvKEY = btrecvKEY.decode('utf-8')
            print("bluetooth client : ", btrecvKEY)

            time.sleep(0.1)

            if btrecvKEY is not None:
                break

            btconnected_socket.send(btrecvKEY)

    except:
        pass

    print(btrecvKEY)
    print("bluetooth socket closed")

    btconnected_socket.close()
    btserver_socket.close()

    svclient_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    svclient_socket.connect((svHOST, svPORT))

    start_new_thread(svrecv_data, (svclient_socket,))
    print("server connected")
    svclient_socket.send('raspi'.encode())

    time.sleep(0.4)

    stop_threads = False


    svclient_socket.close()
    print("server socket closed")




    if btrecvKEY == decryptotp:
        GPIO.output(2, True)

        print("unlocked")

        time.sleep(5)
        GPIO.output(2, False)
        print("locked")
        btrecvKEY = None

    else:
        print("locked")
        btrecvKEY = None

