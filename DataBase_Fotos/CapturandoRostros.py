import cv2
import os
import imutils

personName = 'Sergio'
dataPath = 'C:/Users/sergi/OneDrive/Desktop/DataBase_Fotos/Data'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture("C:/Users/sergi/OneDrive/Desktop/DataBase_Fotos/Sebastian.mp4")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

# Configuración de CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]

        # Convertir a escala de grises
        rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)

        # Realzar el contraste
        alpha = 0.9  # Controla el contraste (1.0 no cambia el contraste)
        beta = 0  # Controla el brillo
        rostro_contrast = cv2.convertScaleAbs(rostro_gray, alpha=alpha, beta=beta)

        # Aplicar realce de contraste adaptativo (CLAHE)
        rostro_contrast_enhanced = clahe.apply(rostro_contrast)
        
        # Ajuste de brillo y contraste local
        rostro_contrast_local = cv2.addWeighted(rostro_contrast_enhanced, 0.9, rostro_contrast_enhanced, 0, 0)

        # Guardar la imagen del rostro con filtro aplicado
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro_contrast_local)
        count = count + 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
