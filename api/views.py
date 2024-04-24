from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import pandas as pd
import numpy as np
import threading
import qrcode
import base64
import pickle
import joblib
import openpyxl
import cv2
import os
import io


def index(request):
    return render(request, "app/index.html")


def bienvenido(request):
    return render(request, 'app/welcome.html')


def aplicate_model(model, frame):
    cwd = os.getcwd()
    excel_directory = os.path.join(
        cwd, 'api/models', 'LBP_Codes.xlsx')
    # LBP_Codes = openpyxl.load_workbook(excel_directory)
    LBP_Codes = pd.read_excel(excel_directory)
    LBP_Codes = np.array(LBP_Codes)

    # Función LBP
    def LBP_Brinez(Img_gris, LBP_Codes):
        [Fl, Cl] = Img_gris.shape  # Almacena el número de filas y Columnas
        Matriz_LBP = np.zeros((Fl, Cl))
        Pesos = np.array([[1, 2, 4], [128, 0, 8], [64, 32, 16]])

        for j in range(1, Cl-1, 1):  # Columnas
            for i in range(1, Fl-1, 1):  # Filas
                Region = Img_gris[i-1:i+2, j-1:j+2]
                Referencia = Img_gris[i, j]
                Region_Bin = Region >= Referencia
                Escalada = Region_Bin*Pesos
                Suma = np.sum(Escalada)
                Codigo = LBP_Codes[Suma, 1]
                Matriz_LBP[i, j] = Codigo

        LBP_Histograna = np.histogram(Matriz_LBP, bins=59, range=(0, 58))
        LBP_Histograna = LBP_Histograna[0]
        LBP_Vector = np.zeros((1, len(LBP_Histograna)))
        for i in range(len(LBP_Histograna)):
            LBP_Vector[0, i] = LBP_Histograna[i]
        return (Matriz_LBP, LBP_Vector)

    Filas = 300
    Columnas = 300
    I_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    I_gris = cv2.resize(I_gris, (Filas, Columnas),
                        interpolation=cv2.INTER_AREA)
    # plt.imshow(frame[:, :, [2, 1, 0]], cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # Aplicando el proceso a la imagen de validación
    Matriz_Características = np.zeros(
        (1, 59*16))  # 16 vectores LBP sin etiqueta
    # Detector = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
    # Cara = Detector.detectMultiScale(I_gris, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20), maxSize=(300, 300))
    Detector = cv2.CascadeClassifier(
        '/haarcascade_frontalface_default.xml')
    I_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Cara = Detector.detectMultiScale(I_gris, 1.1, 5)
    Tamaño_Roi = int(48)
    Filas_2 = Tamaño_Roi
    Columnas_2 = Tamaño_Roi
    Particiones = Filas_2/4

    for (x, y, w, h) in Cara:
        Recorte = I_gris[y:y+h, x:x+w]
        Recorte = cv2.resize(Recorte, (Filas_2, Columnas_2),
                             interpolation=cv2.INTER_AREA)
        # plt.imshow(Recorte.astype('uint8'),cmap='gray',vmin=0, vmax=255)
        # plt.show()

    del Cara

    Contador = 0
    for j in range(0, 4):
        for k in range(0, 4):
            Matriz_LBP, Matriz_Características[0, Contador*59:(Contador*59+59)] = LBP_Brinez(
                Recorte[j*12:(j*12+12), k*12:(k*12 + 12)], LBP_Codes)
            Contador += 1

    Prediccion = model.predict(Matriz_Características)

    Nombres = ["Michael", "Freyner", "Camilo",
               "Viviana", "Daniel", "Yesid", "Alfonso"]

    return Nombres[int(Prediccion[0])-1]


def readModelFace(frame):
    try:
        # Obtener el directorio de trabajo actual
        cwd = os.getcwd()
        current_directory = os.path.join(
            cwd, 'api/models', 'Modelo_faces_KNN.pkl')

        # Cargar archivos desde el sistema local
        with open(current_directory, 'rb') as f:
            modelo_entrenado = joblib.load(f)

        response = aplicate_model(modelo_entrenado, frame)
        return response
    except:
        return 0


def start_camera(request):
    def stop_camera():
        global stop_flag
        stop_flag = True

    def generate_qr_code(frame):
        base64Frame = base64.b64encode(
            cv2.imencode('.jpg', frame)[1]).decode('utf-8')
        img = qrcode.make(base64Frame)
        type(img)
        img.save("qr.png")

        # Leer la imagen desde el archivo
        with open("qr.png", "rb") as img_file:
            # Codificar la imagen como base64
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        return image_data

    def detect_bounding_box(vid):
        cwd = os.getcwd()
        directory = os.path.join(
            cwd, 'models', 'haarcascade_frontalface_default.xml')
        face_classifier = cv2.CascadeClassifier(
            '/haarcascade_frontalface_default.xml')
        I_gris = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(I_gris, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        return faces

    global stop_flag
    stop_flag = False

    # threading.Timer(10.0, stop_camera).start()

    video_capture = cv2.VideoCapture(0)

    while not stop_flag:
        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        # apply the function we created to the video frame
        faces = detect_bounding_box(video_frame)

        # Generar el código QR a partir del frame actual

        scan_result = readModelFace(video_frame)
        print(scan_result)

        if scan_result != 0:
            qr_image_base64 = generate_qr_code(faces)
            threading.Timer(2.0, stop_camera).start()
            # stop_camera()
        else:
            qr_image_base64 = None
            scan_result = 'None'

        cv2.imshow("Camera", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return JsonResponse({'qr': qr_image_base64, 'scan': scan_result})
