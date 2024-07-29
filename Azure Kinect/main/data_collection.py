import argparse

import pandas as pd

import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
from tkinter.messagebox import showinfo
import cv2
import pickle
import time
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='3_12_stand_1.csv', help='文件名')
    parser.add_argument('--time_num', type=int, default=0, help='系统时间')
    parser.add_argument('--wrong_posture_time', type=int, default=0, help='错误坐姿持续时间')
    parser.add_argument('--sitting_time_second', type=int, default=0, help='坐姿时间——秒')
    parser.add_argument('--sitting_time_minute', type=int, default=0, help='坐姿时间——分')
    parser.add_argument('--predict_num', type=int, default=0, help='预测结果')

    opt = parser.parse_args()
    return opt

"""
加载机器学习模型
"""


def load_model():
    f2 = open('svm_RPF.model', 'rb')
    model = f2.read()
    model1 = pickle.loads(model)
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


def calculate_angle_2d(A, B, C):
    # 将点转换为numpy数组
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # 计算向量
    AB = A - B
    BC = C - B

    # 计算两个向量的点乘
    dot_product = np.dot(AB, BC)

    # 计算向量的模
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    # 使用点乘公式和arccos计算角度（以弧度为单位）
    angle = np.arccos(dot_product / (magnitude_AB * magnitude_BC))

    # 将弧度转换为度
    angle_deg = np.degrees(angle)

    return angle_deg



def collect_data(body_info_json, args, model):
    # 收集指定关节点
    neck = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_NECK]['position']
    head = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_HEAD]['position']
#    nose = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_NOSE]['position']
    chest = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SPINE_CHEST]['position']
    shoulder_left = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SHOULDER_LEFT]['position']
    shoulder_right = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SHOULDER_RIGHT]['position']
    clavicle_right = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_CLAVICLE_RIGHT]['position']
    clavicle_left = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_CLAVICLE_LEFT]['position']
    navel = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_SPINE_NAVEL]['position']
    pelvis = body_info_json["skeleton"]["joints"][pykinect.K4ABT_JOINT_PELVIS]['position']

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



    """
    angle_1 = calculate_angle(auxiliary_pos, pelvis_pos, navel_pos)
    angle_2 = calculate_angle(auxiliary_pos, pelvis_pos, chest_pos)
    angle_3 = calculate_angle(pelvis_pos, navel_pos, chest_pos)
    angle_4 = calculate_angle(navel_pos, chest_pos, head_pos)
    angle_5 = calculate_angle(auxiliary_pos, pelvis_pos, neck_pos)
    """

    """
    # with open(args.filename, 'a') as file:
    #     file.write(
    #         str(round(neck['x'], 2)) + ' ,' + str(round(neck['y'], 2)) + ' ,' + str(
    #             round(neck['z'], 2)) + ' ,' + str(round(head['x'], 2)) + ' ,' + str(
    #             round(head['y'], 2)) + ' ,' + str(round(head['z'], 2)) + ' ,' + str(
    #             round(shoulder_right['x'], 2)) + ' ,' + str(
    #             round(shoulder_right['y'], 2)) + ' ,' + str(round(shoulder_right['z'], 2)) + ' ,' + str(
    #             round(shoulder_left['x'], 2)) + ' ,' + str(round(shoulder_left['y'], 2)) + ' ,' + str(
    #             round(shoulder_left['z'], 2)) + ' ,' + str(
    #             round(chest['x'], 2)) + ' ,' + str(round(chest['y'], 2)) + ' ,' + str(
    #             round(chest['z'], 2)) + ' ,' + str(round(navel['x'], 2)) + ' ,' + str(
    #             round(navel['y'], 2)) + ' ,' + str(round(navel['z'], 2)) + ' ,' + str(
    #             round(pelvis['x'], 2)) + ' ,' + str(round(pelvis['y'], 2)) + ' ,' + str(
    #             round(pelvis['z'], 2)) + ' ,' + str(round(nose['x'], 2)) + ' ,' + str(
    #             round(nose['y'], 2)) + ' ,' + str(round(nose['z'], 2))
    #         + ',' + str(round(angle_01_02, 2)) + ' ,' + str(round(angle_01_03, 2))
    #         + ' ,' + str(round(angle_01_04, 2)) + ' ,' + str(round(angle_01_06, 2))
    #         + ' ,' + str(round(angle_40_46, 2)) + ' ,' + str(round(angle_40_42, 2)) + ' ,' + str(
    #             round(angle_40_43, 2)) + '\n')
"""
    # # 需要预测的数据
    # pre_data = [round(neck['x'], 2), round(neck['y'], 2), round(neck['z'], 2),
    #             round(head['x'], 2), round(head['y'], 2), round(head['z'], 2),
    #             round(shoulder_right['x'], 2), round(shoulder_right['y'], 2), round(shoulder_right['z'], 2),
    #             round(shoulder_left['x'], 2), round(shoulder_left['y'], 2), round(shoulder_left['z'], 2),
    #             round(chest['x'], 2), round(chest['y'], 2), round(chest['z'], 2),
    #             round(navel['x'], 2), round(navel['y'], 2), round(navel['z'], 2),
    #             round(pelvis['x'], 2), round(pelvis['y'], 2), round(pelvis['z'], 2),
    #             round(nose['x'], 2), round(nose['y'], 2), round(nose['z'], 2), round(angle_01_02, 2),
    #             round(angle_01_03, 2), round(angle_01_04, 2),
    #             round(angle_01_06, 2), round(angle_40_46, 2), round(angle_40_42, 2),
    #             round(angle_40_43, 2)
    #             ]
    # pre_data = np.array(pre_data).reshape(1, -1)
    # predict_num = model.predict(pd.DataFrame(pre_data))
    # print(predict_num)
    # return predict_num


def predict_posture(predict_num):
    if predict_num == 1:
        res_predict = '健康坐姿'
    elif predict_num == 2:
        res_predict = '身体后倾'
    elif predict_num == 3:
        res_predict = '腰背弯曲'
    elif predict_num == 4:
        res_predict = '身体左倾'
    elif predict_num == 5:
        res_predict = '身体右倾'
    elif predict_num == 6:
        res_predict = '身体前倾'
    elif predict_num == 7:
        res_predict = '站立'
    return res_predict


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
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


def main():
    args = parse_opt()
    pykinect.initialize_libraries(track_body=True)
    # 修改摄像机配置
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # 开启摄像机
    device = pykinect.start_device(config=device_config)
    # 开启身体追踪
    bodyTracker = pykinect.start_body_tracker()
    cv2.namedWindow('Depth image with skeleton', cv2.WINDOW_NORMAL)
    # 加载模型
    model = load_model()
    while True:
        args.predict_num = 0
        capture = device.update()
        sensor_capture = _k4a.k4a_capture_t
        # 获取身体追踪器骨骼
        body_frame = bodyTracker.update()
        ret, depth_color_image = capture.get_colored_depth_image()
        ret, body_image_color = body_frame.get_segmentation_image()
        if not ret:
            continue

        if body_frame.get_num_bodies() <= 0:
            continue

        body_info_json = body_frame.get_body().json()
        time.sleep(0.3)
        args.time_num = args.time_num + 1
        if body_info_json:
            # 收集数据和判断预测结果
            collect_data(body_info_json, args, model)

            # Combine both images
            combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

            # 画出骨骼
            combined_image = body_frame.draw_bodies(combined_image)
            cv2.imshow('Depth image with skeleton', combined_image)

            # 按q退出程序
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
