import numpy

from kalman import *
from yoloV3 import *
from build_utils import utils, img_utils
labelPath = "./yolo-coco/coco.names"
LABELS = open(labelPath).read().strip().split("\n")


# 计算叉乘符号
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_yolo_model(nvidia_gpu='cuda:0'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = Darknet('./yolo-coco/yolov3-spp.cfg').to(device)
    # print(model)
    weights = './yolo-coco/yolov3-spp-ultralytics-416.pt'
    if weights.endswith(".pt") or weights.endswith(".pth"):
        print("加载")
        ckpt = torch.load(weights, map_location=device)
        # load model
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt["model"], strict=False)
    return model


def get_yolo_parameter():
    import json
    line = [(0, 150), (2560, 150)]
    # 车辆总数
    counter = 0
    # 正向车道的车辆数据
    counter_up = 0
    # 逆向车道的车辆数据
    counter_down = 0
    # 创建跟踪器对象
    # tracker = Sort() 原版，没有速度跟踪
    tracker = Sort2()
    memory = {}
    labelPath = "./yolo-coco/coco.names"
    net = get_yolo_model()
    LABELS = open(labelPath).read().strip().split("\n")
    # 生成多种不同的颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    # 获取yolo中每一层的名称
    # 获取输出层的名称: [yolo-82,yolo-94,yolo-106]
    time_backtracking = []
    category_index = {}
    for i in range(len(LABELS)):
        category_index[i + 1] = LABELS[i]
    return net, COLORS, time_backtracking, tracker, memory, category_index


def yolo_predict(frame, net, img_size, W=None, H=None):
    img_o = frame
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # 将图像转换为blob,进行前向传播
    # blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    img = torch.zeros((1, 3, img_size, img_size), device='cuda:0')
    input_size = (img_size, img_size)
    img = img_utils.letterbox(frame, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cuda:0').float()
    img /= 255.0  # scale (0, 255) to (0, 1)
    img = img.unsqueeze(0)  # add batch dimension

    start = time.time()
    # 前向传播，进行预测，返回目标框边界和相应的概率
    with torch.no_grad():

        layerOutputs = net(img)
    end = time.time()

    pred = utils.non_max_suppression(layerOutputs[0], conf_thres=0.5, iou_thres=0.6, multi_label=True)[0]
    t3 = time.time()

    if pred is None:
        print("No target detected.")
        return None
    # process detections
    print(img.shape[2:],img_o.shape,type(img_o.shape))
    pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
    boxes = pred[:, :4].detach().cpu().numpy()
    scores = pred[:, 4].detach().cpu().numpy()
    classes = pred[:, 5].detach().cpu().numpy().astype(np.int)
    pred = pred.detach().cpu().numpy()

    # result = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
    return pred, scores, boxes, classes


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


def bigsort(pred, scores, tracker, memory, boxes, img_o, time_backtracking, COLORS, classes, all_object_paramters):
    dets = []
    if len(pred) > 0:
        for i in range(len(pred)):
            if LABELS[classes[i]] == "car" or LABELS[classes[i]] == 'truck' or LABELS[classes[i]] == 'bus':
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
    for i in range(len(tracks)):
        # print('测试', track.kf.x.T[0][4:])
        boxes.append([tracks[i][0], tracks[i][1], tracks[i][2], tracks[i][3]])
        indexIDs.append(int(tracks[i][4]))
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

    return img_o, memory, time_backtracking, tracker.all_object_parameters.keys()


if __name__ == '__main__':
    vs = cv2.VideoCapture('./video/project_video.mp4')
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
