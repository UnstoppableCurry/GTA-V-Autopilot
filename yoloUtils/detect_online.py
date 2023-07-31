import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from yoloUtils.utils.datasets import LoadStreams, LoadImages
from yoloUtils.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yoloUtils.utils.plots import plot_one_box
from yoloUtils.utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np


def get_model(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_img = False
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def predict(model, img, im0s, names, colors, opt):
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()
    boxes = []
    confidences = []
    classIDs = []
    dets = []
    for i, det in enumerate(pred):  # detections per image
        s, im0, frame = '', im0s, img

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            # print(img.shape[2:],im0.shape,type(im0.shape))

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            #                                   # img                      im0s
            #         detection=det.detach().cpu().numpy()
            classID = det[:, 5].detach().cpu().numpy().astype(np.int)
            confidence = det[:, 4].detach().cpu().numpy()[0]
            # print(classID)
            # if confidence > 0.1: #and (
            #         # names[int(classID[0])] == 'car' or names[int(classID[0])] == 'truck' or names[int(classID)] == 'bus'):
            #     boxes.append(det[:, :4].detach().cpu().numpy())
            #     confidences.append(float(confidence))
            #     classIDs.append(classID)
            #     dets.append(det[:, :5].detach().cpu().numpy())
            for *xyxy, conf, cls in reversed(det):
                if conf > 0.1 and (names[int(cls)] == 'car' or names[int(cls)] == 'truck' or names[int(cls)] == 'bus'):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    cv2.imshow('result', im0)
                    confidences.append(float(conf))
                    classIDs.append(int(cls))
                    dets_input = [float(x) for x in xyxy]
                    dets_input.append(float(conf))
                    dets.append(np.array(dets_input))

            #     cv2.waitKey(1)  # 1 millisecond
    if len(dets) == 0:
        return None
    else:
        return dets, boxes, classIDs


def detect(opt):
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
    t0 = time.time()
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        success, im0s = cap.read()
        img = im0s
        # for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device).permute(2, 0, 1)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        predict(model, img, im0s, names, colors, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view_img', type=bool, default=True)
    parser.add_argument('--source', type=str, default='1234.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='weight/v5lite-s.pt', help='model.pt path(s)')
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
