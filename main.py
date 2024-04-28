import cv2 as cv
from ultralytics import YOLO

# загружаем модель
model = YOLO('best_foraminifera.pt')

# открываем изображение
image_path = 'foraminifera_test.jpg'
frame = cv.imread(image_path)

# проходимся нейросеткой по кадру
results = model(frame)

# визуализируем результат на кадре
annotated_frame = frame.copy()

for r in results:
    for c in r.boxes:
        conf = c.conf.cpu().numpy()[0]
        if conf > 0.6:
            annotated_frame = results[0].plot()

dim = (1920, 1080)
resized_frame = cv.resize(annotated_frame, dim, interpolation=cv.INTER_AREA)

# показываем проанализированный кадр
cv.imshow("Analyzed image", resized_frame)
cv.waitKey(0)
cv.destroyAllWindows()
