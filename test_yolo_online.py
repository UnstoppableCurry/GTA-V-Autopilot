import os, sys

root_path = os.path.dirname((os.path.abspath(__file__)))
print(root_path)
sys.path.append(root_path)
import sys
sys.path.append('yoloUtils')
import models.experimental
import detect_online
import utils
from yoloUtils.detect_online import *
from kalman_test import *


def get_yolo_parameter():
    import json
    line = [(0, 150), (2560, 150)]
    counter = 0
    counter_up = 0
    counter_down = 0
    tracker = Sort2()
    memory = {}
    labelPath = "./utils/coco.names"
    LABELS = open(labelPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    time_backtracking = []
    category_index = {}
    for i in range(len(LABELS)):
        category_index[i + 1] = LABELS[i]
    return COLORS, time_backtracking, tracker, memory, category_index


def run_online_yolo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view_img', type=bool, default=True)
    parser.add_argument('--source', type=str, default='1234.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='yoloUtils/weight/v5lite-s.pt',
                        help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='sample', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view_img', type=bool, default=True)
    parser.add_argument('--source', type=str, default='1234.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='yoloUtils/weight/v5lite-s.pt',
                        help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='sample', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    model = get_model(opt)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # vs = cv2.VideoCapture(0)
    vs=cv2.VideoCapture('project_video.mp4')
    COLORS, time_backtracking, tracker, memory, category_index = get_yolo_parameter()
    all_object_paramters = {}
    while vs.isOpened():
        success, im0s = vs.read()
        # print(im0s.shape)
        img = im0s
        img = cv2.resize(img, (640, 480))
        # for path, img, im0s, vid_cap in dataset:
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img).to(device).permute(2, 0, 1)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        result = predict(model, img, im0s, names, colors, opt)

        if result:
            dets, boxes, classIDs = result
            # print(result)
            # print(type(np.array(dets).reshape(-1, 5)), type(tracker), type(memory), type(boxes),
            #       type(im0s),
            #       type(time_backtracking),
            #       type(COLORS), type(classIDs), type(all_object_paramters))

            if dets is None or boxes is None or classIDs is None:
                continue
            img_o, memory, time_backtracking, all_object_paramters = bigsort(np.array(dets).reshape(-1, 5), tracker,
                                                                             memory, boxes,
                                                                             im0s,
                                                                             time_backtracking,
                                                                             COLORS, classIDs, all_object_paramters,category_index)
            cv2.imshow('frame', img_o)

        else:
            img_o = im0s
            cv2.imshow('frame', img_o)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
