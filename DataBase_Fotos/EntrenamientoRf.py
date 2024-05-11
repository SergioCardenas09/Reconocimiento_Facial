import cv2
import os
import numpy as np

dataPath = 'C:/Users/sergi/OneDrive/Desktop/DataBase_Fotos/Data'
peopleList = [person for person in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, person))]
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for personName in peopleList:
    personPath = os.path.join(dataPath, personName)
    print('Leyendo las imágenes de', personName)

    # Obtener la lista de archivos de imagen válidos en la carpeta de la persona
    imageFiles = [file for file in os.listdir(personPath) if file.endswith(('.jpg', '.jpeg', '.png'))]

    for fileName in imageFiles:
        print('Rostros:', personName + '/' + fileName)
        labels.append(label)
        face = cv2.imread(os.path.join(personPath, fileName), cv2.IMREAD_GRAYSCALE)
        face = cv2.resize(face, (150, 150))
        facesData.append(face)
    label += 1

# Métodos para entrenar el reconocedor
face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('modeloFisherFace2.xml')
print("Modelo almacenado...")
