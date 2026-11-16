import cv2  #Para capturar vídeo de la cámara, leer imágenes, dibujar rectángulos y texto, mostrar la ventana.
import dlib #Librería de machine learning, tiene un detector de caras
import numpy as np   #Numpy es una libreria para utiliza arrays y matrices en vez de numeros.
import os   #para leer las carpetas
import pickle  #para guardar y cargar los encodings

def cargar_encodings_desde_pickle(ruta_pkl="codificaciones_dlib.pkl"):
    """
    Carga los encodings y nombres guardados previamente por entrenamiento_dlib.py.
    Evita procesar todas las fotos cada vez que se inicia la cámara.
    """
    if os.path.exists(ruta_pkl):
        print(f"Cargando base de datos biométrica desde {ruta_pkl}...")
        with open(ruta_pkl, "rb") as f:
            datos = pickle.load(f)
        # Usamos exactamente las claves de tu archivo: "encodings" y "names"
        return datos["encodings"], datos["names"]
    
    print(f"Archivo {ruta_pkl} no encontrado. Se usará el escaneo de carpetas.")
    return [], []


def cargar_encodings_desde_carpetas(ruta_base="imagenes_personas",                                        #donde están las subcarpetas con las fotos.
                                    predictor_path="shape_predictor_68_face_landmarks.dat",               #donde esta el archivo .dat con el modelo de 68 puntos faciales.
                                    face_rec_model_path="dlib_face_recognition_resnet_model_v1.dat"):     #donde esta el archivo .dat con el modelo ResNet que convierte el rostro en un vector.
    """
    Recorre cada subcarpeta en ruta_base, lee todas las imágenes .jpg/.png,
    extrae el descriptor facial (128D) de cada rostro usando dlib.
    Retorna: (lista_descriptores, lista_nombres)
    """
    # Inicializar detector, predictor y modelo de reconocimiento
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    encodings_conocidos = []
    nombres_conocidos = []

    if not os.path.exists(ruta_base):
        print(f"Carpeta '{ruta_base}' no encontrada.")
        return [], []

    for nombre_persona in os.listdir(ruta_base):
        carpeta_persona = os.path.join(ruta_base, nombre_persona)
        if not os.path.isdir(carpeta_persona):
            continue

        print(f"Procesando: {nombre_persona}")
        for archivo in os.listdir(carpeta_persona):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                ruta_img = os.path.join(carpeta_persona, archivo)
                img = cv2.imread(ruta_img)
                if img is None:
                    print(f"No se pudo leer: {archivo}")
                    continue
                # Detectar rostros
                caras = detector(img, 1)
                if len(caras) == 0:
                    print(f"No se detectó rostro en: {archivo}")
                elif len(caras) > 1:
                    print(f"Múltiples rostros en {archivo}, usando solo el primero.")
                    shape = predictor(img, caras[0])
                    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
                    encodings_conocidos.append(np.array(face_descriptor))
                    nombres_conocidos.append(nombre_persona)
                else:
                    shape = predictor(img, caras[0])
                    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
                    encodings_conocidos.append(np.array(face_descriptor))
                    nombres_conocidos.append(nombre_persona)

    print(f"Cargados {len(encodings_conocidos)} rostros de {len(set(nombres_conocidos))} personas.\n")
    return encodings_conocidos, nombres_conocidos


# Inicializar detector dlib para la cámara (solo detección, no el modelo completo cada frame)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


frame_skip = 0                          #Para saltarte frames y mejorar rendimiento
PROCESAR_CADA_N_FRAMES = 1              # =


#Se metio el bucle while dentro de la funcion debido a que es necesario para ejecutarlo en interfaz.py
def iniciar_camara_web(encodings_conocidos, nombres_conocidos):

    #Se usa global para poder usar las variables
    global frame_skip, PROCESAR_CADA_N_FRAMES, detector, predictor, face_rec_model

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_skip += 1
        if frame_skip % PROCESAR_CADA_N_FRAMES != 0:
            cv2.imshow("Reconocimiento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Reduci el tamaño para que dlib sea mas rapido.
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detectar rostros en la imagen reducida (dlib detecta en RGB)
        caras = detector(rgb_small, 1)

        for cara in caras:
            # Obtener puntos de referencia y descriptor facial
            shape = predictor(rgb_small, cara)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_small, shape)
            face_descriptor = np.array(face_descriptor)

            # Comparar con todos los descriptores conocidos
            if len(encodings_conocidos) > 0:
                # Calcular distancia euclídea
                distancias = np.linalg.norm(encodings_conocidos - face_descriptor, axis=1)
                mejor_indice = np.argmin(distancias)
                umbral = 0.6  # Ajusta según pruebas (menor = más exigente)
                nombre = "Desconocido"
                if distancias[mejor_indice] < umbral:
                    nombre = nombres_conocidos[mejor_indice]
            else:
                nombre = "Desconocido"

            # Escalar coordenadas al tamaño original
            left = cara.left() * 2
            top = cara.top() * 2
            right = cara.right() * 2
            bottom = cara.bottom() * 2

            # Dibujar
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



#Esto es una medida de seguridad para que en la interfaz no se ejecute el entrenamiento cada vez que se importe el módulo
#Sino que se ejecuta solo cuando se llame explícitamente a la función desde el botón de la interfaz
#Tambien añádi por si acaso el inf len == 0
if __name__ == "__main__":
    encodings_conocidos, nombres_conocidos = cargar_encodings_desde_carpetas()
    if len(encodings_conocidos) == 0:
        print("No se encontraron rostros. Asegúrate de tener subcarpetas dentro de 'imagenes_personas'.")
        exit()

    iniciar_camara_web(encodings_conocidos, nombres_conocidos)
