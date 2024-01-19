import cv2
import time

# Constantes
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
CONFIG_FILE = "yolov4-tiny.cfg"
WEIGHTS_FILE = "yolov4-tiny.weights"
CLASS_NAMES_FILE = "coco.names"
VIDEO_FILE = "cam1_sub3_sprint.mp4"

def load_classes(file_path):
    with open(file_path, "r") as f:
        return [cname.strip() for cname in f.readlines()]

def setup_detection_model(config_file, weights_file):
    net = cv2.dnn.readNet(weights_file, config_file)
    model = cv2.dnn.DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    return model

def show_coordinates(frame, classes, scores, boxes, class_names):
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]}: {score:.2f}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Mostra as coordenadas na tela
        x, y, w, h = box
        coordinates_label = f"Coordenadas: ({x}, {y}), Largura: {w}, Altura: {h}"
        cv2.putText(frame, coordinates_label, (box[0], box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Carrega classes
    print('loading data')
    class_names = load_classes(CLASS_NAMES_FILE)

    # Captura o vídeo
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Configura o modelo de detecção
    model = setup_detection_model(CONFIG_FILE, WEIGHTS_FILE)

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_AREA)

        start = time.time()
        classes, scores, boxes = model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
        end = time.time()

        show_coordinates(frame, classes, scores, boxes, class_names)

        fps_label = f"FPS: {round(1.0 / (end - start), 2)}"

        # Escreve o FPS na imagem
        cv2.putText(frame, fps_label, (0, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("detections", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
