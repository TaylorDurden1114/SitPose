import argparse
import os
import tkinter as tk
import threading
from collections import Counter
import platform
from tkinter import font, ttk
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from pykinect_azure.k4a import _k4a
import pykinect_azure as pykinect
from tkinter.messagebox import showinfo
import cv2
import pickle
import time
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont, ImageTk
from joblib import load




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prolonged_sitting_count', type=int, default=0, help='久坐次数')
    parser.add_argument('--wrong_posture_count', type=int, default=0)
    parser.add_argument('--wrong_postures_list', type=int, default=[0]*8)
    parser.add_argument('--maximum_sitting_time', type=int, default=0, help='最长久坐时间')
    parser.add_argument('--avg_sitting_time', type=int, default=0, help='最长久坐时间')
    parser.add_argument('--filename', type=str, default='12_14_h_1.csv', help='文件名')
    parser.add_argument('--time_num', type=int, default=0, help='System time')
    parser.add_argument('--wrong_posture_time', type=int, default=0, help='错误坐姿持续时间')
    parser.add_argument('--hourly_sitting_time', type=int, default=[0] * 24, help='每小时的坐姿时间')
    parser.add_argument('--sitting_time_all', type=int, default=0, help='all time')
    parser.add_argument('--sitting_time_minute', type=int, default=0, help='坐姿时间——分')
    parser.add_argument('--predict_num', type=int, default=0, help='预测结果')
    parser.add_argument('--predict_posture', type=str, default='', help='predict result')
    parser.add_argument('--posture_array', nargs='*', type=int, default=[0] * 60,
                        help='A list of values for sitting posture')
    opt = parser.parse_args()
    return opt


class AppData:
    def __init__(self):
        self.window = None
        self.label_text = None


app_data = AppData()


def load_model():
    # f2 = open('voting_classifier.joblib', 'rb')
    # model = f2.read()
    # model1 = pickle.loads(model)
    model1 = load('voting_classifier.joblib')
    return model1

def calculate_angle(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    AB = A - B
    BC = C - B

    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    angle = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
    angle_deg = np.degrees(angle)

    return angle_deg

def collect_data(body_info_json, args, model):
    time1 = time.time()
    # 收集指定关节点
    neck = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_NECK]['position']
    head = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_HEAD]['position']
    nose = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_NOSE]['position']
    chest = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SPINE_CHEST]['position']
    shoulder_left = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SHOULDER_LEFT]['position']
    shoulder_right = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SHOULDER_RIGHT]['position']
    navel = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SPINE_NAVEL]['position']
    pelvis = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_PELVIS]['position']
    clavicle_right = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_CLAVICLE_RIGHT]['position']
    clavicle_left = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_CLAVICLE_LEFT]['position']
    navel
    time2 = time.time()


    neck_pos = (neck['x'], neck['y'], neck['z'])
    head_pos = (head['x'], head['y'], head['z'])
    chest_pos = (chest['x'], chest['y'], chest['z'])
    navel_pos = (navel['x'], navel['y'], navel['z'])
    pelvis_pos = (pelvis['x'], pelvis['y'], pelvis['z'])
    shoulder_right_pos = (shoulder_right['x'], shoulder_right['y'], shoulder_right['z'])
    shoulder_left_pos = (shoulder_left['x'], shoulder_left['y'], shoulder_left['z'])
    clavicle_right_pos = (clavicle_right['x'], clavicle_right['y'], clavicle_right['z'])
    clavicle_left_pos = (clavicle_left['x'], clavicle_left['y'], clavicle_left['z'])
    auxiliary_pelvis_pos = (pelvis['x'] + 100, pelvis['y'], pelvis['z'])
    auxiliary_clavicle_right_pos = (clavicle_right['x'] + 100, clavicle_right['y'], clavicle_right['z'])
    auxiliary_clavicle_left_pos = (clavicle_left['x'] + 100, clavicle_left['y'], clavicle_left['z'])
    auxiliary_neck_pos = (neck['x'] + 100, neck['y'], neck['z'])


    # 计算角度并添加到列表中
    angle_1 = calculate_angle(auxiliary_pelvis_pos, pelvis_pos, navel_pos)
    angle_2 = calculate_angle(auxiliary_pelvis_pos, pelvis_pos, chest_pos)
    angle_3 = calculate_angle(pelvis_pos, navel_pos, chest_pos)
    angle_4 = calculate_angle(navel_pos, chest_pos, head_pos)
    angle_5 = calculate_angle(auxiliary_pelvis_pos, pelvis_pos, neck_pos)
    angle_6 = calculate_angle(auxiliary_neck_pos, neck_pos, head_pos)
    angle_7 = calculate_angle(auxiliary_clavicle_right_pos, clavicle_right_pos, shoulder_right_pos)
    angle_8 = calculate_angle(auxiliary_clavicle_left_pos, clavicle_left_pos, shoulder_left_pos)
    angle_9 = calculate_angle(shoulder_right_pos, clavicle_right_pos, chest_pos)
    angle_10 = calculate_angle(shoulder_left_pos, clavicle_left_pos, chest_pos)

    # 写入文件中
    with open(args.filename, 'a') as file:
        file.write(
            str(str(round(head['x'], 2)) + ' ,' + str(round(head['y'], 2)) + ' ,' + str(round(head['z'], 2)) + ' ,' +
                str(round(angle_1, 2)) + ' ,' + str(round(angle_2, 2)) + ' ,' + str(round(angle_3, 2)) + ' ,' +
                str(round(angle_4, 2)) + ' ,' + str(round(angle_5, 2)) + ' ,' + str(round(angle_6, 2)) + ' ,' +
                str(round(angle_7, 2)) + ' ,' + str(round(angle_8, 2)) + ' ,' + str(round(angle_9, 2)) + ' ,' +
                str(round(angle_10, 2)) + '\n'))

    # 需要预测的数据
    pre_data = [round(head['x'], 2), round(head['y'], 2), round(head['z'], 2),
                round(angle_1, 2), round(angle_2, 2),  round(angle_3, 2),
                round(angle_4, 2),  round(angle_5, 2),  round(angle_6, 2),
                round(angle_7, 2),  round(angle_8, 2), round(angle_9, 2), round(angle_10, 2)
                ]
    pre_data = np.array(pre_data).reshape(1, -1)
    predict_num = model.predict(pre_data)
    print(predict_num)
    return predict_num


def predict_posture(predict_num):
    # 确保 predict_num 是一个标量值
    if isinstance(predict_num, np.ndarray) and predict_num.size == 1:
        predict_num = predict_num.item()

    posture_dict = {
        1: 'Sitting straight',
        2: 'Lying',
        3: 'Hunched over',
        4: 'Left sitting',
        5: 'Right sitting',
        6: 'Lean forward',
        7: 'Standing'
    }
    return posture_dict.get(predict_num, "Unknown posture")


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "msyh.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def update_gui(image, window, predict_posture, label_text):
    font_posture = font.Font(family='Microsoft YaHei', size=30)
    photo = ImageTk.PhotoImage(image=image)
    image_label = tk.Label(window, image=photo)
    image_label.place(x=0, y=200)
    # 保存对 photo 的引用，防止它被垃圾收集器回收
    image_label.image = photo

    label_text.set(predict_posture)


def process_kinect_data(args, device, bodyTracker, model, window, label_text):
    while True:
        args.predict_num = 0
        capture = device.update()
        sensor_capture = _k4a.k4a_capture_t
        # 获取身体追踪器骨骼
        body_frame = bodyTracker.update()
        ret, depth_color_image = capture.get_colored_depth_image()
        ret, body_image_color = body_frame.get_segmentation_image()
        time.sleep(0.8)
        if not ret:
            continue

        if body_frame.get_num_bodies() <= 0:
            args.predict_posture = 'none'
            sitting_is_true = False
#            continue
        else:
            body_info_json = body_frame.get_body().json()
            sitting_is_true = True


        # 开启时间
        args.time_num = args.time_num + 1
        if sitting_is_true:
            args.predict_num = collect_data(body_info_json, args, model)
            args.predict_posture = predict_posture(args.predict_num)
            combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
            # 画出骨骼
            # combined_image = body_frame.draw_bodies(combined_image)
            # combined_image = cv2ImgAddText(combined_image, args.predict_posture, 100, 40, (255, 0, 255), 50)
            # image_pil = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
            # monitor_sitting_posture(args)
            # window.after(0, update_gui, image_pil, window, args.predict_posture, label_text)
        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        if body_frame.get_num_bodies() > 0:
            combined_image = body_frame.draw_bodies(combined_image)
            monitor_sitting_posture(args)
        combined_image = cv2ImgAddText(combined_image, args.predict_posture, 100, 40, (255, 0, 255), 50)
        image_pil = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        window.after(0, update_gui, image_pil, window, args.predict_posture, label_text)


def setup_gui(args):
    # 修改args.model
    # args.model = 'new_model_value'
    # 关闭初始窗口并打开主GUI

    window = tk.Tk()
    window.title("Sitting Posture Detection System")
    window.geometry("900x700")

    # 设置主题
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except tk.TclError:
        pass

    # 设置字体
    myFont = font.Font(family='Microsoft YaHei', size=30, weight='bold')
    myFont_depth_img = font.Font(family='Microsoft YaHei', size=20)
    font_posture = font.Font(family='Microsoft YaHei', size=30)

    # 添加标题和标签
    tk.Label(window, text="Sitting Posture Detection System", font=myFont).place(x=150, y=15)
    tk.Label(window, text="Depth img", font=myFont_depth_img).place(x=150, y=150)
    tk.Label(window, text="Current sitting posture", font=myFont_depth_img).place(x=550, y=150)

    label_text = tk.StringVar()
    label_text.set("Start")
    label_posture = tk.Label(window, textvariable=label_text, font=font_posture, fg='red')
    label_posture.place(x=560, y=240)

    # 添加按钮
    tk.Button(window, text="Sitting Posture Report", font=("Microsoft YaHei", 12), width=30, height=5,
              command=lambda: posture_report(args, window)).place(x=550,y=350)
    tk.Button(window, text="Sedentary Report", font=("Microsoft YaHei", 12), width=30, height=5,
              command=lambda: sedentary_report(args, window)).place(x=550, y=500)

    return window, label_text


def open_main_gui(args, window, selected_option):
    # 清理主窗口上的所有控件
    for widget in window.winfo_children():
        widget.destroy()

    if selected_option == "option1":
        print("model1")
    elif selected_option == "option2":
        print("model2")
    elif selected_option == "option3":
        print("model3")
    else:
        print("默认")

    # 设置字体
    myFont = font.Font(family='Microsoft YaHei', size=30, weight='bold')
    myFont_depth_img = font.Font(family='Microsoft YaHei', size=20)
    font_posture = font.Font(family='Microsoft YaHei', size=30)

    # 添加标题和标签
    tk.Label(window, text="Sitting Posture Detection System", font=myFont).place(x=150, y=15)
    tk.Label(window, text="Depth img", font=myFont_depth_img).place(x=150, y=150)
    tk.Label(window, text="Current sitting posture", font=myFont_depth_img).place(x=550, y=150)

    label_text = tk.StringVar()
    label_text.set("Start")
    label_posture = tk.Label(window, textvariable=label_text, font=font_posture, fg='red')
    label_posture.place(x=560, y=240)

    # 添加按钮
    tk.Button(window, text="Sitting Posture Report", font=("Microsoft YaHei", 12), width=30, height=5,
              command=lambda: posture_report(args, window)).place(x=550, y=350)
    tk.Button(window, text="Sedentary Report", font=("Microsoft YaHei", 12), width=30, height=5,
              command=lambda: sedentary_report(args, window)).place(x=550, y=500)
    app_data.window = window
    app_data.label_text = label_text

    device, body_tracker = setup_kinect()
    # 加载模型
    model = load_model()

    # 多线程处理深度数据
    thread = threading.Thread(target=start_processing_thread,
                              args=(args, device, body_tracker, model, window, label_text))
    thread.start()

    return window, label_text


def setup_initial_gui(args):
    def confirm():
        selected_option = option_var.get()
        open_main_gui(args, initial_window, selected_option)

    initial_window = tk.Tk()
    initial_window.title("Sitting Posture Detection System")
    initial_window.geometry("900x700")

    # Option variables
    option_var = tk.StringVar(value="none")
    tk.Label(initial_window, text="Please select the camera position", font=("Microsoft YaHei", 35)).place(x=100, y=150)

    # Create checkbuttons
    radiobutton1 = tk.Radiobutton(initial_window, text="Right",
                                  variable=option_var, value="option1", font=("Microsoft YaHei", 25))
    radiobutton1.place(x=170, y=300, width=150, height=50)

    radiobutton2 = tk.Radiobutton(initial_window, text="Left",
                                  variable=option_var, value="option2", font=("Microsoft YaHei", 25))
    radiobutton2.place(x=390, y=300, width=150, height=50)

    radiobutton3 = tk.Radiobutton(initial_window, text="ground",
                                  variable=option_var, value="option3", font=("Microsoft YaHei", 25))
    radiobutton3.place(x=620, y=300, width=150, height=50)

    # Confirm button
    confirm_button = tk.Button(initial_window, text="Confirm",
                               command=lambda: confirm(), font=("Microsoft YaHei", 25))
    confirm_button.place(x=350, y=400, width=250, height=150)
    app_data.window = initial_window
    return initial_window


def setup_kinect():
    pykinect.initialize_libraries(track_body=True)
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device = pykinect.start_device(config=device_config)
    body_tracker = pykinect.start_body_tracker()
    return device, body_tracker


def start_processing_thread(args, device, bodyTracker, model, window, label_text):
    thread = threading.Thread(target=process_kinect_data, args=(args, device, bodyTracker, model, window, label_text))
    thread.daemon = True
    thread.start()


def play_sound():
    if platform.system() == "Windows":
        # 在 Windows 上播放默认警告声音
        import winsound
        try:
            # 尝试播放特定的警告声音
            winsound.MessageBeep(winsound.MB_ICONWARNING)
        except AttributeError:
            # 如果 MB_ICONWARNING 不可用，则播放默认声音
            winsound.MessageBeep()
    else:
        # 在 MacOS 或 Linux 上尝试使用系统命令播放声音
        os.system('afplay /System/Library/Sounds/Glass.aiff')


def create_warning_window():
    warning_window = Toplevel()
    warning_window.title("警告")
    warning_window.attributes('-topmost', True)  # 使窗口始终位于顶部

    warning_window.geometry("150x100")
    warning_label = tk.Label(warning_window, text="你正处于不良坐姿状态")
    warning_label.pack()
    close_button = tk.Button(warning_window, text="确定", command=warning_window.destroy)
    close_button.pack()
    # 播放声音
    play_sound()

    # 更新窗口，以便能够获取其尺寸
    warning_window.update_idletasks()

    # 计算居中位置
    window_width = warning_window.winfo_width()
    window_height = warning_window.winfo_height()
    screen_width = warning_window.winfo_screenwidth()
    screen_height = warning_window.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # 将窗口置于屏幕中央
    warning_window.geometry(f'+{center_x}+{center_y}')


def monitor_sitting_posture(args):
    # 每经过一分钟，将时间清零，将坐姿缓冲区清空
    if args.time_num >= 60:
        args.time_num = 0
        args.posture_list = args.posture_array
        counts = Counter(args.posture_array)
        most_common = counts.most_common(1)
        straight_count = args.posture_array.count(1)
        stand_count = args.posture_array.count(7)
        most_wrong_posture = most_common[0][0]
        # 错误坐姿判断
        if straight_count + stand_count < 40:
            # 把相应的错误坐姿的类型累加
            args.wrong_postures_list[most_wrong_posture] += 1
            # 把错误坐姿总数累加
            args.wrong_posture_count += 1
            create_warning_window()
        args.posture_array = [0] * 60

        # 久坐判断
        if stand_count < 40:
            # 累加当前坐姿时间 和 总共坐姿时间
            args.sitting_time_minute = args.sitting_time_minute + 1
            args.sitting_time_all = args.sitting_time_all + 1
        else:
            args.sitting_time_minute = 0

    # 如果坐姿时间在1小时以上，弹窗警告
    if args.sitting_time_minute >= 60:
        # 久坐次数累加
        args.prolonged_sitting_count = args.prolonged_sitting_count + 1
        result = showinfo('警告', '你正处于久坐状态')
        print(f'警告:{result}')

    if 0 <= args.time_num < len(args.posture_array):
        args.posture_array[args.time_num] = int(args.predict_num)
        print(f"Updated array: {args.posture_array}")
    else:
        print(f"Index out of range. Array length is {len(args.posture_array)}.")

    if args.maximum_sitting_time < args.sitting_time_minute:
        args.maximum_sitting_time = args.sitting_time_minute


def sedentary_report(args, window):
    # 示例数据
    max_sitting_duration = args.maximum_sitting_time
    prolonged_sitting_count = args.prolonged_sitting_count
    total_sitting_time = args.sitting_time_all
    if prolonged_sitting_count != 0:
        args.avg_sitting_time = args.sitting_time_all / args.prolonged_sitting_count
    else:
        args.avg_sitting_time = 0

    # 数据和标签
    values = [max_sitting_duration, args.avg_sitting_time, total_sitting_time]
    labels = ['Max sitting duration', 'Avg Sitting Time', 'Total Sitting Time']

    # 创建条形图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, values, color=['lightskyblue', 'palegreen', 'lightcoral'], width=0.3)
    ax.set_ylabel('Minute')
    ax.set_ylim(bottom=0)  # This line ensures that the y-axis starts from 0
    ax.set_yticks([0, 5, 10, 15])
    # 在新窗口中显示图表
    new_window = Toplevel(window)
    new_window.title("Sedentary Report")

    # 添加标题标签
    title_label = tk.Label(new_window, text="Sedentary Report", font=("Arial", 16))
    # pady 为垂直方向的外边距
    title_label.pack(pady=10)

    # 添加久坐次数的可变标签
    sitting_count_label = tk.Label(new_window, text=f"Prolonged Sitting Count: {args.prolonged_sitting_count}",
                                   font=("Arial", 12))
    sitting_count_label.pack()

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # 添加标题和标签
    plt.ylabel('Min')

def posture_report(args, window):
    wrong_posture_count = args.wrong_posture_count
    wrong_postures_list = args.wrong_postures_list

    # 数据和标签
    values = [wrong_posture_count,  wrong_postures_list[2], wrong_postures_list[3],
              wrong_postures_list[4], wrong_postures_list[5], wrong_postures_list[6], wrong_postures_list[7]]
    labels = ['Wrong posture count',  'Lying', 'Hunched over', 'Left sitting',
              'Right sitting', 'Lean forward', 'Standing']

    # 创建条形图
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, values, color=['lightskyblue',  'palegreen', 'lightcoral',
                                  'silver', 'moccasin', 'mediumslateblue', 'palegoldenrod'], width=0.7)
    ax.set_ylabel('Count', fontsize=18)
    ax.set_xticklabels(labels, rotation=30)

    # 在新窗口中显示图表
    new_window = Toplevel(window)
    new_window.title("Sitting Posture Report")
    ax.set_ylim(bottom=0)
    # ax.set_yticks([0, 5, 10])

    # 添加标题标签
    title_label = tk.Label(new_window, text="Sitting Posture Report", font=("Arial", 16))
    # pady 为垂直方向的外边距
    title_label.pack(pady=10)

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


def main():
    # 加载参数
    args = parse_opt()
    # 初始化坐姿监测系统的GUI
    # setup_initial_gui(args)
    setup_initial_gui(args)
    # window, label_text = open_main_gui(args, initial_window)

    # 事件循环结束后，获取 window 和 label_text
    window = app_data.window
    # label_text = app_data.label_text

    # 初始化Kinect相机
    # device, body_tracker = setup_kinect()
    # 加载模型
    # model = load_model()
    # 多线程处理深度数据
#     start_processing_thread(args, device, body_tracker, model, window, label_text)
    window.mainloop()


if __name__ == "__main__":
    main()
