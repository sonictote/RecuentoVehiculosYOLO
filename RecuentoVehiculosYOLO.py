from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Definicion de la region de la linea
REGION_POINTS1 = [(400, 500), (900, 500)] #(eje x, eje y)

# Configura el tipo de fuente, tamaño y COLOR del texto
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255) # blanco
THICKNESS = 2

#Cargamos el modelo
model = YOLO("yolov8n.pt") 

# Capturar video
cap = cv2.VideoCapture("ejercicio 5-5-YOLO\\recursos 5-5-YOLO\\cars.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Iniciamos el contador de objetos
counter1 = object_counter.ObjectCounter()

#Definimos los argumentos de la clase counter
counter1.set_args(view_img=False,
                 reg_pts=REGION_POINTS1,
                 classes_names=model.names,
                 draw_tracks=True,
                 view_in_counts=False,
                 view_out_counts=False)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Escribe el texto en el frame
    tracks = model.track(im0, persist=True, show=False)
    counter1.start_counting(im0, tracks)
    
    cv2.putText(im0, f'Entrada: {len(counter1.counting_list)}', (50, 50), FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)
    cv2.imshow('Video', im0)

     # Presiona 'q' para salir
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
