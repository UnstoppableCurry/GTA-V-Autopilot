import torch
from 交通.Autopilot.acc定速循环.yoloV3 import *


def get_yolo_model(nvidia_gpu='cuda:0'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = Darknet('./yolov3-spp.cfg').to(device)
    print(model)
    weights = './yolov3-spp-ultralytics-416.pt'
    if weights.endswith(".pt") or weights.endswith(".pth"):
        print("加载")
        ckpt = torch.load(weights, map_location=device)
        print(ckpt)
        # load model
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt["model"], strict=False)
    return model
