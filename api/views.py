from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import threading
import qrcode
import base64
import cv2
import os
import io


def index(request):
    return render(request, "app/index.html")


def bienvenido(request):
    return render(request, 'app/welcome.html')


# def createQR(request):
#     texto = "Hola, mundo!"
#     qr_code = qrcode.create(texto)

#     # Generar la imagen QR en formato PNG como un objeto BytesIO
#     buffer = io.BytesIO()
#     qr_code.png(buffer, scale=5, module_size=15)
#     buffer.seek(0)

#     # Devolver la imagen como respuesta HTTP
#     return HttpResponse(buffer, content_type="image/png")

def start_camera(request):

    def stop_camera():
        global stop_flag
        stop_flag = True

    def generate_qr_code(frame):
        img = qrcode.make(frame)
        type(img)
        img.save("michael.png")

        return img

    def detect_bounding_box(vid):
        face_classifier = cv2.CascadeClassifier(
            '/haarcascade_frontalface_default.xml')
        I_gris = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(I_gris, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        return faces

    global stop_flag
    stop_flag = False

    threading.Timer(10.0, stop_camera).start()

    video_capture = cv2.VideoCapture(0)

    while not stop_flag:
        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        # apply the function we created to the video frame
        faces = detect_bounding_box(video_frame)

        # Generar el código QR a partir del frame actual
        qr_image = generate_qr_code(faces)

        # display the processed frame in a window named "My Face Detection Project"
        cv2.imshow("My Face Detection Project", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return JsonResponse({'redirect_url': 'http://127.0.0.1:8000/welcome/'})
