import cv2
import numpy as np
import torch
video1_path = 'D:\\videoFile\\32.31.250.103\\20240501_20240501125647_20240501140806_125649.mp4'
import time
import csv

normal_lane_count = 0
emergency_lane_count = 0
boundary_line = 500

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频时长（秒）
    duration = total_frames / fps

    print(f"视频总帧数: {total_frames}")
    print(f"视频帧率: {fps}")
    print(f"视频时长（秒）: {duration}")

    cap.release()


def pic_test(normal_lane_count=normal_lane_count, emergency_lane_count=emergency_lane_count):
    # 定义识别线线的起点和终点坐标
    line_start = np.array([100, 250])  # 起点
    line_end = np.array([1000, 250])  # 终点

    # 水平线的y值，表示车辆是否通过
    horizontal_line_y = 500

    # 初始化参数
    normal_lane_total_count = 0  # 正常车道通过车辆总数
    emergency_lane_total_count = 0  # 应急车道通过车辆总数
    tracked_vehicles = {}

    counter = 0  # 计数器
    buffer_tracks = {}  # 保存位置和时间的元组
    crossed_boxes = {}  # 用于存储每个框的交叉状态的字典
    identification_range = 30  # 识别范围
    counter_id = 0  # 分配唯一 ID 的计数器
    previous_positions = {}

    # 视频的时间加速倍数
    time_scale_factor = 10  # 视频是10倍速

    # 时间和速度的设置
    fps = 25  # 视频实际帧率为25帧每秒
    scaling_factor = 0.05  # 像素到实际距离的缩放比例 (1像素 = 0.05米)
    road_length_km = 0.5  # 道路长度的估计，假设为0.5公里

    # 车辆总数和流量的时间跟踪
    start_time = time.time()

    with open('speed.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_num', 'speed_kmh'])

        cap = cv2.VideoCapture(video1_path)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_num += 1
            if not ret:
                break

            # 定义斜线的两个端点坐标 (x1, y1) 和 (x2, y2)，假设这条线划分了应急车道和普通车道
            point1 = (104, 604)  # 起点
            point2 = (476, 328)  # 终点
            # 计算分界线的斜率
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])  # (y2 - y1) / (x2 - x1)

            # 计算分界线的截距：y = slope * x + intercept
            intercept = point1[1] - slope * point1[0]

            # 绘制分界线
            cv2.line(frame, point1, point2, (0, 255, 0), 2)
            cv2.line(frame, (0, horizontal_line_y), (frame.shape[1], horizontal_line_y), (0, 0, 255), 2)

            # 使用YOLOv5模型检测车辆
            results = model(frame)
            detected_boxes = results.xyxy[0].numpy()  # 获取检测框

            # 车辆检测后的处理
            for box in detected_boxes:
                x1, y1, x2, y2, confidence, cls = box  # 获取检测框的坐标和类别
                center = np.array([box[2], box[3]], dtype=int)  # 使用车辆右下角作为中心点

                box_id = None

                # 在缓冲区中寻找之前的车辆ID
                for id, past_positions in buffer_tracks.items():
                    if len(past_positions) > 0 and np.linalg.norm(past_positions[-1][0] - center) < identification_range:
                        box_id = id
                        break

                if box_id is None:
                    # 为新检测到的车辆分配唯一 ID
                    box_id = counter_id
                    counter_id += 1
                    buffer_tracks[box_id] = []
                    crossed_boxes[box_id] = False
                #
                buffer_tracks[box_id].append((center, time.time()))  # 记录位置和时间
                previous_positions[box_id] = buffer_tracks[box_id][-1][0]

                # 计算车速（修正计算）
                speed_kph = 0  # 初始化速度为0
                if box_id in previous_positions and len(buffer_tracks[box_id]) > 1:
                    prev_x, prev_y = previous_positions[box_id]
                    displacement_pixels = np.linalg.norm(np.array([center[0] - buffer_tracks[box_id][-2][0][0], center[1] - buffer_tracks[box_id][-2][0][1]]))
                    print(center[0],buffer_tracks[box_id][-2][0][0])
                    # 限定位置
                    if buffer_tracks[box_id][-2][0][1]>300 and center[1] < 300:
                        if displacement_pixels > 0:  # 避免速度为0的情况
                            displacement_meters = displacement_pixels * scaling_factor
                            speed_mps = displacement_meters * fps * time_scale_factor  # 速度：米/秒 (调整为实际速度)
                            speed_mps = displacement_meters * 25
                            speed_kph = speed_mps * 3.6  # 转换为公里/小时
                            # 写入表中  frame_num | speed_kph
                            row = [frame_num,speed_kph]
                            writer.writerow(row)

                # 在车辆检测框上显示速度
                cv2.putText(frame, f'Speed: {speed_kph:.2f} km/h', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2)

                # 检查是否越过水平线
                if len(buffer_tracks[box_id]) > 1:
                    if (buffer_tracks[box_id][-2][0][1] - line_start[1]) * (
                            buffer_tracks[box_id][-1][0][1] - line_start[1]) < 0:
                        # 车辆越过了水平线，记录车道
                        center_x = buffer_tracks[box_id][-1][0][0]
                        center_y = buffer_tracks[box_id][-1][0][1]
                        line_y_at_center_x = slope * center_x + intercept  # 分界线在 center_x 位置的 y 值
                        if center_y < line_y_at_center_x:  # 在普通车道
                            normal_lane_total_count += 1
                        else:  # 在应急车道
                            emergency_lane_total_count += 1
                        counter += 1

                # 绘制车辆检测框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 计算车流密度：检测到的车辆数量 / 道路长度
            vehicle_density = len(buffer_tracks) / road_length_km

            # 计算车流量（每小时通过虚拟线的车辆数）
            elapsed_time = time.time() - start_time  # 计算经过的时间
            if elapsed_time > 0:
                vehicle_flow = (counter / elapsed_time) * 3600  # 将流量转换为每小时的值
            else:
                vehicle_flow = 0

            # 显示普通车道和应急车道的总数
            cv2.putText(frame, f"Normal Lane: {normal_lane_total_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            cv2.putText(frame, f"Emergency Lane: {emergency_lane_total_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

            # 显示车流量和车流密度
            cv2.putText(frame, f"Vehicle Flow (per hour): {vehicle_flow:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicle Density (vehicles/km): {vehicle_density:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0), 2)

            # 显示当前帧
            cv2.imshow('YOLOv5 Detection', frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# def mouse_event(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         print(f"Coordinates: ({x}, {y})")
#
#         # 在窗口标题中显示当前鼠标的坐标
#         img_copy = image.copy()
#         cv2.putText(img_copy, f"Coordinates: ({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         cv2.imshow('Image with Coordinates', img_copy)


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None

    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率: {fps} FPS")
    return fps

# 鼠标点击事件的回调函数
def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在鼠标左键按下时，打印点击的坐标点
        print(f"鼠标点击坐标: ({x}, {y})")


# 主函数，获取视频第一帧并绑定鼠标事件
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return

    # 显示第一帧
    cv2.imshow('First Frame', frame)

    # 绑定鼠标点击事件到窗口
    cv2.setMouseCallback('First Frame', mouse_click_event)

    # 等待用户按 'q' 键退出
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def get_scale_mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # 当左键按下时，记录点
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 在点的位置绘制一个圆
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)  # 连接两个点的线
            pixel_distance = ((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2) ** 0.5
            print(f"像素距离: {pixel_distance:.2f} 像素")
            points = []  # 清空点列表，准备重新计算




if __name__ == '__main__':
    # image = cv2.imread("C:\\Users\\79295\Desktop\\QQ20240921-091306.png")
    #
    # # 创建一个副本来显示坐标
    # image_copy = image.copy()
    #
    # # 创建窗口
    # cv2.namedWindow('Image with Coordinates')
    #
    # # 绑定鼠标事件
    # cv2.setMouseCallback('Image with Coordinates', mouse_event)
    #
    # # 显示图片，并且鼠标悬停时显示坐标
    # while True:
    #     # 在窗口中显示有坐标的图像
    #     cv2.imshow('Image with Coordinates', image_copy)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
    #         break
    #
    # # 释放窗口
    # cv2.destroyAllWindows()
    # # pic_test()
    # 获取视频帧率
    # fps = get_video_fps(video1_path)
    #
    # # 处理视频，显示第一帧并捕获鼠标点击事件
    # process_video(video1_path)

    # pic_test()
    # get_video_info(video1_path)
    # pic_test()

    ## 计算两点之间像素
    # cap = cv2.VideoCapture(video1_path)
    #
    # # 读取视频的第一帧
    # ret, frame = cap.read()
    #
    # if not ret:
    #     print("无法读取视频帧")
    #     cap.release()
    #     exit()
    #
    # # 回调函数，用于鼠标点击获取两个点并计算距离
    # points = []
    #
    #
    # cv2.imshow("Frame", frame)
    #
    # # 显示视频的第一帧
    # cv2.imshow("Frame", frame)
    # cv2.setMouseCallback("Frame", get_scale_mouse_callback)
    #
    # # 等待用户按键退出
    # cv2.waitKey(0)
    # cap.release()
    # cv2.destroyAllWindows()

    pic_test()