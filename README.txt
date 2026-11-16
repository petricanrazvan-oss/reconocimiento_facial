dlib usa HOG y SVM para detectar caras. utiliza 68 puntos estos siendo:

Puntos 1 al 17: El contorno de la mandíbula y el mentón.
Puntos 18 al 27: Las cejas.
Puntos 28 al 36: El puente y la base de la nariz.
Puntos 37 al 48: Los ojos.
Puntos 49 al 68: Los labios y la boca.

Los utiliza de plantilla para poder reconocer caras.

Tambien utiliza un modelo de RESTNET. Este ha sido entrenado por miles de caras. Esto lo hace capaz de generar unos valores especificos para,
calcular caras. Como tal el programa no mira pixeles ni narices, sino mira vectores y desde ellos es capaz de generar 128 numeros que es como 
un codigo de barras para caras. Esto hace que al sacar diferentes imagenes puedas tener unos valores casi identicos, pero al utilizar a otra 
persona tenga resultados completamente distintos.

HOG sirve para simplificar la imagen para que el pc pueda entenderla. Lo que hace HOG es coger los pixeles y generar lineas matematicamente
perfectas.Esto ocurre porque mira la diferencia de color brillo y demas entre los pixeles y decide como y que lineas genrar. El programa conoce caras
asi que es capaz de ver los ojos (que son mas oscuros que la piel) y diferenciarlos de la frente. esto hace que pueda diferenciar las caras sin importar
si alguien es de piel oscura o si hay mucha o poca luz en la habitación.

SVM es el algoritmo de clasificacion. Este programa ha sido entrenado con miles de dibujos de HOG de caras y de otras cosas que no son caras, como arboles.
De esta forma SVM es capaz de crear una linea divisoria (el cuadrado) de la cara y el background. Esto lo hace con cada frame de la imagen, despues de que HOG
le haya mandado la imagen procesada. 

Asi que dlib usa HOG para extraer las 
siluetas y formas clave de una foto, 
y luego se las pasa a SVM para que este 
confirme si esa forma corresponde o no a 
un rostro humano.

