import customtkinter as ctk
import entrenamiento_dlib
import reconocimiento
import os
import cv2
from PIL import Image, ImageTk

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")

class AppReconocimiento(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Reconocimiento Facial IA")
        self.geometry("1280x720") # Ampliamos el ancho para que quepa la cámara a la derecha
        self.resizable(False, False)

        # Variables de control de la cámara
        self.cap = None
        self.camara_activa = False
        self.encodings_conocidos = []
        self.nombres_conocidos = []

        # --- CONTENEDOR IZQUIERDO (PANEL DE CONTROL) ---
        self.panel_izquierdo = ctk.CTkFrame(self, width=350)
        self.panel_izquierdo.pack(side="left", fill="y", padx=15, pady=15)

        self.titulo = ctk.CTkLabel(self.panel_izquierdo, text="PANEL DE CONTROL IA", font=ctk.CTkFont(size=18, weight="bold"))
        self.titulo.pack(pady=15)

        self.lbl_nombre = ctk.CTkLabel(self.panel_izquierdo, text="Nombre para registrar usuario:", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_nombre.pack(pady=5)

        self.txt_nombre = ctk.CTkEntry(self.panel_izquierdo, placeholder_text="Escribe el nombre aquí...", width=250)
        self.txt_nombre.pack(pady=5)

        self.btn_capturar = ctk.CTkButton(self.panel_izquierdo, text="Capturar Rostro desde Cámara", command=self.ejecutar_captura, fg_color="#3A3A3A", hover_color="#808080")
        self.btn_capturar.pack(pady=5)

        self.separador = ctk.CTkLabel(self.panel_izquierdo, text="-----------------------------------------", text_color="gray")
        self.separador.pack(pady=5)

        self.btn_entrenar = ctk.CTkButton(self.panel_izquierdo, text="1. Reentrenar Carpetas", command=self.ejecutar_entrenamiento, height=35)
        self.btn_entrenar.pack(pady=10, fill="x", padx=20)

        self.btn_reconocimiento = ctk.CTkButton(self.panel_izquierdo, text="2. Iniciar Reconocimiento", command=self.ejecutar_reconocimiento, height=35, fg_color="#228B22", hover_color="#006400")
        self.btn_reconocimiento.pack(pady=5, fill="x", padx=20)

        # ¡NUEVO BOTÓN! Para detener el proceso de forma normal y limpia
        self.btn_detener = ctk.CTkButton(self.panel_izquierdo, text="3. Detener Cámara", command=self.detener_camara, height=35, fg_color="#B22222", hover_color="#8B0000", state="disabled")
        self.btn_detener.pack(pady=5, fill="x", padx=20)

        self.estado = ctk.CTkLabel(self.panel_izquierdo, text="Sistema listo. Esperando acción...", font=ctk.CTkFont(size=12), wraplength=280)
        self.estado.pack(side="bottom", pady=15)

        # --- CONTENEDOR DERECHO (PANTALLA DE LA CÁMARA) ---
        self.panel_derecho = ctk.CTkFrame(self, width=500, height=450, fg_color="black")
        self.panel_derecho.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        # Aquí es donde se dibujará el vídeo en tiempo real dentro de la app
        self.lbl_camara = ctk.CTkLabel(self.panel_derecho, text="CÁMARA APAGADA", text_color="white", font=ctk.CTkFont(size=14))
        self.lbl_camara.pack(expand=True)

    def ejecutar_captura(self):
        nombre = self.txt_nombre.get()
        if not nombre.strip():
            self.estado.configure(text="Error: Escribe un nombre.")
            return

        if self.camara_activa:
            self.detener_camara() # Apagamos el reconocimiento si estaba puesto

        self.estado.configure(text=f"Capturando fotos para {nombre}...")
        self.update()
        
        exito = entrenamiento_dlib.capturar_nuevo_usuario_desde_camara(nombre, 0)
        if exito:
            self.estado.configure(text=f"Usuario {nombre} registrado correctamente.")
            self.txt_nombre.delete(0, 'end')
        else:
            self.estado.configure(text="Error o captura cancelada.")

    def ejecutar_entrenamiento(self):
        self.estado.configure(text="Procesando subcarpetas... Espere.")
        self.update()
        exito = entrenamiento_dlib.guardar_entrenamiento_en_disco()
        if exito:
            self.estado.configure(text="¡Base de datos .PKL generada!")
        else:
            self.estado.configure(text="Error: Añade fotos a las subcarpetas.")

    def ejecutar_reconocimiento(self):
        
        if self.camara_activa:
            return # Ya está encendida

        self.estado.configure(text="Cargando base de datos biométrica...")
        self.update()
        
        # Leemos el archivo .pkl (tarda milisegundos). No escanea carpetas ni entrena.
        self.encodings_conocidos, self.nombres_conocidos = reconocimiento.cargar_encodings_desde_pickle()
        
        #Si el .pkl no existe, solo entonces escanea las carpetas
        if len(self.encodings_conocidos) == 0:
            self.estado.configure(text="PKL no encontrado. Escaneando carpetas de respaldo...")
            self.update()
            self.encodings_conocidos, self.nombres_conocidos = reconocimiento.cargar_encodings_desde_carpetas()
        
        # Si sigue vacío tras el plan de emergencia, es que de verdad no hay datos
        if len(self.encodings_conocidos) == 0:
            self.estado.configure(text="Error: No hay datos guardados para reconocer.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.estado.configure(text="Error: No se pudo abrir la cámara.")
            return

        # Activamos los controles correctos
        self.camara_activa = True
        self.btn_detener.configure(state="normal")
        self.btn_reconocimiento.configure(state="disabled")
        self.estado.configure(text="Cámara integrada encendida con éxito.")
        
        # Iniciamos el bucle visual integrado
        self.actualizar_frame()

    def actualizar_frame(self):
        
        if not self.camara_activa:
            return

        ret, frame = self.cap.read()
        if ret:
            reconocimiento.frame_skip += 1
            
            # Solo procesamos la IA cada N frames para mantener altos los FPS
            if reconocimiento.frame_skip % reconocimiento.PROCESAR_CADA_N_FRAMES == 0:
                reconocimiento.frame_skip = 0
                
                # Reducimos tamaño para que dlib vuele
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                caras = reconocimiento.detector(rgb_small, 1)

                for cara in caras:
                    shape = reconocimiento.predictor(rgb_small, cara)
                    face_descriptor = reconocimiento.face_rec_model.compute_face_descriptor(rgb_small, shape)
                    face_descriptor = os.sys.modules['numpy'].array(face_descriptor)

                    nombre = "Desconocido"
                    texto_pantalla = "Desconocido"
                    color_recuadro = (0, 0, 255) # Rojo para desconocidos

                    if len(self.encodings_conocidos) > 0:
                        distancias = os.sys.modules['numpy'].linalg.norm(self.encodings_conocidos - face_descriptor, axis=1)
                        mejor_indice = distancias.argmin()
                        distancia_minima = distancias[mejor_indice]
                        umbral_maximo =   0.6                      
                        if distancia_minima < umbral_maximo:
                            nombre = self.nombres_conocidos[mejor_indice]
                            color_recuadro = (0, 255, 0) # Verde para conocidos
                            
                            # --- CÁLCULO DE PROBABILIDAD (NUEVO) ---
                            # Si distancia es 0.0 -> 100% prob. Si es 0.6 -> 0% prob respecto al umbral.
                            probabilidad = (1.0 - (distancia_minima / umbral_maximo) + 0.35) * 100
                            # Aseguramos que el porcentaje no de números locos por encima de 100 o debajo de 0
                            probabilidad = max(0, min(100, probabilidad))
                            
                            texto_pantalla = f"{nombre} ({probabilidad:.1f}%)"
                        else:
                            # Si detecta a alguien pero está lejos del umbral, calcula una prob genérica baja
                            probabilidad = max(0, (1.0 - distancia_minima) * 100)
                            texto_pantalla = f"Desconocido ({probabilidad:.1f}%)"
                    else:
                        texto_pantalla = "Desconocido"

                    left, top, right, bottom = cara.left()*2, cara.top()*2, cara.right()*2, cara.bottom()*2
                    
                    # Dibujamos el rectángulo alrededor del rostro
                    cv2.rectangle(frame, (left, top), (right, bottom), color_recuadro, 2)
                    
                    # Ponemos el texto con el nombre y el porcentaje encima de la cabeza
                    cv2.putText(frame, texto_pantalla, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_recuadro, 2)

            # Convertimos el frame de OpenCV (BGR) a Formato de Interfaz (RGB -> PIL Image -> CTkImage)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cv2image)
            img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(500, 410))
            
            self.lbl_camara.configure(image=img_ctk, text="")
            self.lbl_camara.image = img_ctk 

        self.after(10, self.actualizar_frame)

    def detener_camara(self):
        """Termina el proceso de la cámara de manera normal, controlada y limpia"""
        self.camara_activa = False
        if self.cap:
            self.cap.release()
        
        # Reset de la interfaz
        self.lbl_camara.configure(image=None, text="CÁMARA APAGADA")
        self.btn_reconocimiento.configure(state="normal")
        self.estado.configure(text="Cámara cerrada de forma segura.")

        self.after(1000, self.limpiar_panel_camara)
    
    def limpiar_panel_camara(self):
        """Función auxiliar que restablece el fondo negro del panel visual"""
        self.lbl_camara.configure(image=None, text="CÁMARA APAGADA")
        self.lbl_camara.image = None # Limpiamos la referencia de memoria de Pillow
        self.estado.configure(text="Cámara cerrada de forma segura.")

    def on_closing(self):
        """Seguridad por si cierran la ventana directamente desde la 'X' de Windows"""
        self.detener_camara()
        self.destroy()
    

if __name__ == "__main__":
    app = AppReconocimiento()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Captura el botón cerrar de la ventana
    app.mainloop()
