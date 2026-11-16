import dlib
import cv2
import numpy as np
import pickle
import os

detector = dlib.get_frontal_face_detector()

# Rutas a los modelos (ya están dentro de tu venv)
# ESTO ESTA COMENTADO PARA PODER USARLO EN MI PROPIO PC
# DESCOMENTARLO PARA QUE FUNCIONE EN EL TUYO Y LUEGO COMENTAR MI PARTE
#model_path = "venv/lib/python3.12/site-packages/face_recognition_models/models/"
#predictor = dlib.shape_predictor(model_path + "shape_predictor_68_face_landmarks.dat")
#facerec = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
 


#Lo mismo que en el reconcocimiento.py pero con el bucle while dentro de la función para poder usarlo desde la interfaz
#Tambien se agrego la capacidad de poder guardar las fotos en subcarpetas
def guardar_entrenamiento_en_disco():
    PATH_BASE = "imagenes_personas"
    known_encodings = []
    known_names = []

    if not os.path.exists(PATH_BASE):
        print(f"Carpeta '{PATH_BASE}' no encontrada. Creándola automáticamente...")
        os.makedirs(PATH_BASE)
        return False

    # 1. Recorremos cada subcarpeta (cada subcarpeta representa a UNA persona)
    for nombre_persona in os.listdir(PATH_BASE):
        carpeta_persona = os.path.join(PATH_BASE, nombre_persona)
        
        # Nos aseguramos de saltarnos archivos sueltos y leer solo carpetas
        if not os.path.isdir(carpeta_persona):
            continue

        print(f"Procesando imágenes de: {nombre_persona}")
        
        # 2. Leemos todas las fotos que haya DENTRO de la carpeta de esa persona
        for archivo in os.listdir(carpeta_persona):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_completa = os.path.join(carpeta_persona, archivo)
                img = cv2.imread(ruta_completa)

                if img is None:
                    print(f"No se pudo leer {archivo}")
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dets = detector(rgb, 1)

                if len(dets) == 0:
                    print(f"No se detectó rostro en {archivo}")
                    continue

                # Extraemos las características del rostro
                shape = predictor(rgb, dets[0])
                encoding = facerec.compute_face_descriptor(rgb, shape)
                
                # Guardamos los datos asociándolos al nombre de la CARPETA, no del archivo
                known_encodings.append(np.array(encoding))
                known_names.append(nombre_persona)
                print(f"Foto entrenada con éxito: {archivo}")

    if len(known_encodings) == 0:
        print("No se encontraron rostros válidos para guardar.")
        return False

    # 3. Guardamos la estructura en la base de datos binaria .pkl
    with open("codificaciones_dlib.pkl", "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"Entrenamiento completo. {len(known_encodings)} rostros guardados en 'codificaciones_dlib.pkl'")
    return True

def capturar_nuevo_usuario_desde_camara(nombre_usuario, indice_camara=0):
    """
    Abre la cámara para tomar fotos. Evita ráfagas repetidas controlando 
    la pulsación de la barra espaciadora para ahorrar espacio.
    """
    PATH_BASE = "imagenes_personas"
    carpeta_usuario = os.path.join(PATH_BASE, nombre_usuario)
    
    if not os.path.exists(carpeta_usuario):
        os.makedirs(carpeta_usuario)
        
    cap = cv2.VideoCapture(indice_camara)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara para registrar.")
        return False
        
    contador_fotos = 0
    limite_fotos = 3  # Tomamos solo 3 fotos en total
    print(f"Capturando {limite_fotos} fotos para {nombre_usuario}.")
    
    while contador_fotos < limite_fotos:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Redimensionamos el frame a un tamaño estándar para que no pese megabytes
        frame = cv2.resize(frame, (640, 480))
            
        frame_visual = frame.copy()
        cv2.putText(frame_visual, f"Capturando foto {contador_fotos + 1}/{limite_fotos}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_visual, "Presiona ESPACIO para tomar foto", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Registro de Usuario", frame_visual)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(rgb, 1)
            
            if len(dets) == 1:
                ruta_foto = os.path.join(carpeta_usuario, f"{nombre_usuario}_{contador_fotos}.jpg")
                cv2.imwrite(ruta_foto, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                print(f"Foto {contador_fotos + 1} guardada.")
                
                contador_fotos += 1
                
            
                # Mostramos un cartel de "Procesando..." y pausamos el código 1 segundo entero.
                # Así le da tiempo de sobra a tu dedo a soltar la barra espaciadora.
                frame_pausa = frame_visual.copy()
                cv2.putText(frame_pausa, "¡FOTO CAPTURADA! Soltar tecla...", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.imshow("Registro de Usuario", frame_pausa)
                cv2.waitKey(1000)  # Pausa exacta de 1000 milisegundos (1 segundo)
                
            elif len(dets) == 0:
                print("No se detecta ningún rostro. Inténtalo de nuevo.")
            else:
                print("¡Hay más de un rostro! Asegúrate de estar solo.")
                
        elif key == ord('q'):
            print("Captura cancelada.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if contador_fotos == limite_fotos:
        return guardar_entrenamiento_en_disco()
    return False


#Esto es una medida de seguridad para que en la interfaz no se ejecute el entrenamiento cada vez que se importe el módulo
#Sino que se ejecuta solo cuando se llame explícitamente a la función desde el botón de la interfaz
if __name__ == "__main__":
    guardar_entrenamiento_en_disco()