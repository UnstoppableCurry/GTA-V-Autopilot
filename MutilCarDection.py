from kalman import *
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 计算叉乘符号
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_yolo_parameter():
    line = [(0, 150), (2560, 150)]
    # 车辆总数
    counter = 0
    # 正向车道的车辆数据
    counter_up = 0
    # 逆向车道的车辆数据
    counter_down = 0
    # 创建跟踪器对象
    tracker = Sort()
    memory = {}
    labelPath = "./yolo-coco/coco.names"
    # 加载预训练的模型：权重 配置信息,进行恢复
    weightsPath = "./yolo-coco/yolov3.weights"
    configPath = "./yolo-coco/yolov3.cfg"
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    LABELS = open(labelPath).read().strip().split("\n")
    # 生成多种不同的颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    # 获取yolo中每一层的名称
    ln = net.getLayerNames()
    # 获取输出层的名称: [yolo-82,yolo-94,yolo-106]
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    time_backtracking = []
    return net, COLORS, time_backtracking, tracker, ln, memory


def yolo_predict(frame, net, COLORS, time_backtracking, tracker, ln, memory, W=None, H=None):
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # 将图像转换为blob,进行前向传播
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # 将blob送入网络
    net.setInput(blob)
    start = time.time()
    # 前向传播，进行预测，返回目标框边界和相应的概率
    layerOutputs = net.forward(ln)
    end = time.time()
    print(end - start)
    # 存放目标的检测框
    boxes = []
    # 置信度
    confidences = []
    # 目标类别
    classIDs = []
    # 遍历每个输出
    for output in layerOutputs:
        # 遍历检测结果
        for detection in output:
            # detction:1*85 [5:]表示类别，[0:4]bbox的位置信息 【5】置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.3:
                # 将检测结果与原图片进行适配
                box = detection[0:4] * np.array([W, H, W, H])
                box[box < 0] = 0
                if box[0] > W:
                    box[0] = W
                if box[2] > W:
                    box[2] = W
                if box[1] > H:
                    box[1] = W
                if box[3] > W:
                    box[3] = W
                (centerX, centerY, width, height) = box.astype("int")
                # 左上角坐标
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                # 更新目标框，置信度，类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 非极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 检测框:左上角和右下角
    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # if LABELS[classIDs[i]] == "car":
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            dets.append([x, y, x + w, y + h, confidences[i]])
    # 类型设置
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    # # 显示
    # plt.imshow(frame[:,:,::-1])
    # plt.show()

    # SORT目标跟踪
    time2 = time.time()
    if np.size(dets) == 0:
        return None
    else:
        tracks = tracker.update(dets)
    # 跟踪框
    boxes = []
    # 置信度
    indexIDs = []
    # 前一帧跟踪结果
    previous = memory.copy()
    if len(time_backtracking) < 20:
        time_backtracking.append(previous)
    else:
        time_backtracking.pop(0)
        time_backtracking.append(previous)
    memory = {}
    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    # 碰撞检测
    if len(boxes) > 0:
        i = int(0)
        # 遍历跟踪框
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]  # 对方框的颜色进行设定
            cv2.rectangle(frame, (x, y), (w, h), color, 2)  # 将方框绘制在画面上
            time_previous = [i for i in time_backtracking]
            p0_time = None
            index_time_back = 0
            for time_back in time_previous:
                if indexIDs[i] in time_back:
                    index_time_back += 1
                    previous_box = time_back[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    if None == p0_time:
                        p0_time = p1
                        cv2.line(frame, p1, p1, color, 1)
                    else:
                        cv2.line(frame, p1, p0_time, color, 1)
                        p0_time = p1
            i += 1
    return frame


if __name__ == '__main__':
    vs = cv2.VideoCapture('./video/project_video.mp4')
    net, COLORS, time_backtracking, tracker, ln, memory = get_yolo_parameter()
    while vs.isOpened():
        ret, frame = vs.read()  # 获取每一帧图像
        if ret:
            yolo_predict(frame, net, COLORS, time_backtracking, tracker, ln, memory)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
