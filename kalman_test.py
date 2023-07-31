import numpy
import cv2
from utils.kalman import *
import time

# 计算叉乘符号
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)







def sort(pred, scores, tracker, memory, boxes, img_o, time_backtracking, COLORS, classes):
    dets = []
    if len(pred) > 0:
        for i in range(len(pred)):
            # if LABELS[classIDs[i]] == "car":
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            dets.append([x, y, w, h, scores[i]])
    # 类型设置
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
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
    i = int(0)

    # 碰撞检测
    if len(boxes) > 0:
        # 遍历跟踪框
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]  # 对方框的颜色进行设定
            cv2.rectangle(img_o, (x, y), (w, h), color, 2)  # 将方框绘制在画面上
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
                        cv2.line(img_o, p1, p1, color, 5)
                    else:
                        cv2.line(img_o, p1, p0_time, color, 5)
                        p0_time = p1
            i += 1
    return img_o, memory, time_backtracking


def bigsort(dets, tracker, memory, boxes, img_o, time_backtracking, COLORS, classes, all_object_paramters,category_index):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
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
    for i in range(len(tracks)):
        # print('测试', track.kf.x.T[0][4:])
        boxes.append([tracks[i][0], tracks[i][1], tracks[i][2], tracks[i][3]])
        indexIDs.append(int(tracks[i][4]))
        memory[indexIDs[-1]] = boxes[-1]

    i = int(0)

    # 碰撞检测
    if len(boxes) > 0:
        # 遍历跟踪框
        for index in range(len(boxes)):
            box = boxes[index]
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]  # 对方框的颜色进行设定
            cv2.rectangle(img_o, (x, y), (w, h), color, 2)  # 将方框绘制在画面上
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
                        cv2.line(img_o, p1, p1, color, 5)
                    else:
                        cv2.line(img_o, p1, p0_time, color, 5)
                        p0_time = p1
            i += 1

    return img_o, memory, time_backtracking, tracker.all_object_parameters.keys()


if __name__ == '__main__':
    vs = cv2.VideoCapture('project_video.mp4')
    net, COLORS, time_backtracking, tracker, memory, category_index = get_yolo_parameter()
    net.eval()
    all_object_paramters = {}
    while vs.isOpened():
        ret, frame = vs.read()  # 获取每一帧图像
        if ret:
            result = yolo_predict(frame, net, img_size=608)
            if result:
                pred, scores, boxes, classes = result
                img_o, memory, time_backtracking, all_object_paramters = bigsort(pred, scores, tracker, memory, boxes,
                                                                                 frame,
                                                                                 time_backtracking,
                                                                                 COLORS, classes, all_object_paramters)
            else:
                img_o = frame
            cv2.imshow('frame', img_o)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
