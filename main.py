import cv2
import numpy as np
import torch
video1_path = 'D:\\videoFile\\32.31.250.103\\20240501_20240501125647_20240501140806_125649.mp4'
import time


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
    prev_time = time.time()  # 帧计数器

    # 定义识别线线的起点和终点坐标
    line_start = np.array([100, 250])  # 起点
    line_end = np.array([1000, 250])  # 终点

    counter = 0  # 计数器
    counter_id = 0  # 分配唯一 ID 的计数器
    buffer_tracks = {}  # 保存位置和时间的元组
    crossed_boxes = {}  # 用于存储每个框的交叉状态的字典
    identification_range = 30  # 识别范围

    max_track_length = 35  # 最大轨迹长度
    track_timeout = 10  # 轨迹超时时间（秒）

    unupdated_timeout = 0.1  # 未更新的轨迹存在时间（秒）
    last_updated = {}  # 初始化 last_updated 字典


    cap = cv2.VideoCapture(video1_path)
    # 水平线的y值，表示车辆是否通过
    horizontal_line_y = 500  # 假设横轴的y值为400，用作判断车辆是否通过
    # 存储已通过的车辆ID
    tracked_vehicles = {}
    # 车道车辆计数器
    normal_lane_total_count = 0
    emergency_lane_total_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 定义斜线的两个端点坐标 (x1, y1) 和 (x2, y2)，假设这条线划分了应急车道和正常车道
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
            x1, y1, x2, y2, confidence, cls = box  # 获取边界框的坐标和类别
            # center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=int)
            center = np.array( [box[2], box[3]], dtype=int)

            cv2.circle(frame, tuple(center), radius=5, color=(0, 255, 255), thickness=-1)

            cv2.line(frame, tuple(line_start), tuple(line_end), color=(255, 0, 255), thickness=6)

            # 检查框 ID 是否已存在于缓冲区中
            box_id = None
            for id, past_positions in buffer_tracks.items():
                if len(past_positions) > 0 and np.linalg.norm(
                        past_positions[-1][0] - center) < identification_range:  # 如果 y 位置接近缓冲区中的 1
                    box_id = id
                    break
            '''
            这段代码是用于判断当前的识别框（box）是否在之前的帧中已经识别过。
            在这段代码中，buffer_y_positions是一个字典，它的键是识别框的ID，值是该识别框在过去的帧中所有的y坐标位置。
            这段代码的目标是尝试找到一个已经存在于buffer_y_positions的识别框ID，使得该识别框在上一帧的y坐标位置与当前识别框的y坐标位置非常接近（差距小于30像素）。如果找到了这样的ID，则认为当前的识别框在之前的帧中已经识别过，于是将box_id设置为这个ID，并跳出循环。
            这种做法的目的是尽可能地追踪每一个识别框的运动，即使在连续的帧中，由于摄像设备的振动或者物体的移动速度等原因，相同的物体可能被检测出稍微不同的识别框。通过将新的识别框和旧的识别框相关联，我们可以对物体的运动进行追踪，例如判断物体何时穿过了一条特定的线。
            '''

            if box_id is None:
                # 为盒子分配一个新的 ID
                box_id = counter_id
                counter_id += 1
                buffer_tracks[box_id] = []
                crossed_boxes[box_id] = False  # 将交叉状态初始化为 False

            buffer_tracks[box_id].append((center, time.time()))  # 添加位置和当前时间
            last_updated[box_id] = time.time()  # 更新时间

            # 检查框是否越界
            if len(buffer_tracks[box_id]) > 1:
                if (buffer_tracks[box_id][-2][0][1] - line_start[1]) * (
                        buffer_tracks[box_id][-1][0][1] - line_start[1]) < 0:
                    # 盒子已经越界了
                    crossed_boxes[box_id] = True  # 更新交叉状态为 True
                    center_x = buffer_tracks[box_id][-1][0][0]
                    center_y = buffer_tracks[box_id][-1][0][1]
                    line_y_at_center_x = slope * center_x + intercept  # 分界线在center_x位置的y值
                    if center_y < line_y_at_center_x:  # 在普通车道
                        normal_lane_total_count += 1
                    else:  # 在应急车道
                        emergency_lane_total_count += 1
                    counter +=1
                    print(buffer_tracks[box_id])

            # 如果框被标记为已越线，则绘制文本
            if crossed_boxes[box_id]:
                cv2.putText(frame, 'crossed', (center[0], center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            # # 获取车辆的中心x坐标
            # center_x = (x1 + x2) / 2
            # center_y = (y1 + y2) / 2
            # # 判断车辆是在应急车道还是正常车道
            # # 判断在斜线的上方还是下方
            # # 检查车辆是否已经被跟踪
            # vehicle_id = f"{int(center_x)}_{int(center_y)}"  # 创建车辆的唯一ID（可以基于位置）
            # # 判断车辆是否通过水平线
            # if vehicle_id not in tracked_vehicles and center_y > horizontal_line_y:
            #     # 判断车辆是在应急车道还是普通车道
            #     line_y_at_center_x = slope * center_x + intercept  # 分界线在center_x位置的y值
            #     if center_y < line_y_at_center_x:  # 在普通车道
            #         normal_lane_total_count += 1
            #         tracked_vehicles[vehicle_id] = 'normal'  # 记录该车辆已经通过
            #     else:  # 在应急车道
            #         emergency_lane_total_count += 1
            #         tracked_vehicles[vehicle_id] = 'emergency'
            #
            # # 绘制车辆检测框
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.putText(frame, f'Vehicle {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            #             (255, 255, 255), 2)

            # 显示车道车辆总计数


        for id, track in list(buffer_tracks.items()):
            if len(track) > max_track_length:  # 如果轨迹过长
                track.pop(0)  # 移除最旧的位置
            if time.time() - track[-1][1] > track_timeout:  # 如果轨迹过期
                del buffer_tracks[id]  # 删除整个轨迹

        for id in list(buffer_tracks.keys()):
            if time.time() - last_updated[id] > unupdated_timeout:  # 如果轨迹过期
                del buffer_tracks[id]  # 删除整个轨迹
                del last_updated[id]  # 删除对应的时间


        counter_str = 'Counter: %s' % counter
        cv2.putText(frame, counter_str, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 139, 34), 2)  # 将计数器放在框架上

        cv2.putText(frame, f"Normal Lane Total: {normal_lane_total_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.putText(frame, f"Emergency Lane Total: {emergency_lane_total_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0), 2)

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
    pic_test()