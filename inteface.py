import cv2 as cv
from ultralytics import YOLO
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import psycopg2

# Словарь имен классов, для связи с базой
class_names = {0: 'Chernyshinella disputabilis',
               1: 'Quasiendothyra communis',
               2: 'Priscoidella priscoidea',
               3: 'Aljutovella ajutovica',
               4: 'Quasiendothyra kobeitusana',
               5: 'Uralodiscus rotundus',
               6: 'Fusulinella subpulchra',
               7: 'Quasiendothyra bella',
               8: 'Pseudostafella antiqua',
               9: 'Monotaxinoides transitorius',}

# Загрузка модели
model = YOLO('thin_section.pt')


# Подключение к базе данных PostgreSQL
def connect_to_db():
    return psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="admin")


def get_age(class_name):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT system, otdel, yarus FROM ages WHERE class_name = %s", (class_name,))
    age_info = cursor.fetchone()
    conn.close()
    return age_info if age_info else ("Неизвестно", "Неизвестно", "Неизвестно", "Неизвестно")


def load_image():
    global class_name
    file_path = filedialog.askopenfilename(title="Select an image",
                                           filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
    if file_path:
        frame = cv.imread(file_path)
        results = model(frame)
        annotated_frame = frame.copy()
        max_conf = 0
        class_name = None

        for r in results:
            for c in r.boxes:
                conf = c.conf.cpu().numpy()[0]
                if conf > 0.6 and conf > max_conf:
                    annotated_frame = results[0].plot()
                    max_conf = conf
                    class_id = c.cls.cpu().numpy()[0]
                    class_name = class_names.get(class_id, 'Нет данных')

        if class_name:
            system, otdel, yarus = get_age(class_name)
            label_age.config(text=f"Система: {system}\nОтдел: {otdel}\nЯрус: {yarus}", anchor="w")

        else:
            label_age.config(text="Возраст: Неизвестен")

        dim = (640, 480)
        resized_frame = cv.resize(annotated_frame, dim, interpolation=cv.INTER_AREA)
        img = Image.fromarray(cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        label_image.config(image=img_tk)
        label_image.image = img_tk
    else:
        messagebox.showinfo("Info", "No image selected")


root = Tk()
root.title("Определение возраста с помощью нейросети")
root.geometry("800x600")
root.configure(bg='white')

style_button = {'font': ('Helvetica', 12, 'bold'), 'bg': '#7FB5B5', 'fg': 'white', 'padx': 10, 'pady': 10}
style_label = {'font': ('Helvetica', 12, 'bold'), 'bg': 'white', 'fg': 'black', 'padx': 10, 'pady': 5}

frame_top = Frame(root, bg='white')
frame_top.pack(side=TOP, fill=X)

btn_load = Button(frame_top, text="Загрузить картинку", command=load_image, **style_button)
btn_load.pack(side=LEFT, padx=(20, 10), pady=50)

label_age = Label(frame_top, text="Возраст: ", anchor="w", **style_label)
label_age.pack(side=LEFT, padx=(10, 20), pady=20)

label_image = Label(root, **style_label)
label_image.pack(expand=True, fill=BOTH)

root.mainloop()