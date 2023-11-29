#carrega as dependencias 
import cv2
import time 

#cores das classes
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#carrega as classes 
class_names = []
with open("coco.names", "r") as f:
  class_names = [cname.strip() for cname in f.readlines()]

#captura os videos 
cap = cv2.VideoCapture("vs Tigas - Feito com o Clipchamp_1672530713611")

#carrega os pesos da rede neural 
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") 

#setando os parametros da rn
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#lendo os frames do video 
while True:

  #captura o frame 
  _, frame = cap.read()
  print(frame)

  #começo da contagem
  start = time.time()

  #detecção 
  classes, scores, boxes = model.detect(frame, 0.1, 0.2)

  #fim da contagem 
  end = time.time()

  #Verificar todas as detecções 
  for (classid, score, box) in zip(classes, scores, boxes):

    #dando cor para a classe 
    color = COLORS[int(classid) % len(COLORS)]

    #pegando a classe pelo id e score 
    label = f"{class_names[classid]} : {score}"

    #desenhando a caixinha de detecção 
    cv2.rectangle(frame, box, color, 2)

    #escreve o nome da classe na caixinha 
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  #calcula o tempo q levou pra fazer a detecção 
  fps_label = f"FPS: {round((1.0/(end - start)) ,2)}" 

  #escreve o fps na imagem 
  cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
  cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

  #mostrando a imagem 
  cv2.imshow("detections", frame)
  
  #espera da resposta
  if cv2.waitKey(1) == 27:
    break

# libera camera e destroi janelas 
cap.release()
cv2.destroyAllWindows()
