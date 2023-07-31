# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms
import networks
import cv2



def test_simple(model_name):
    """Function to predict for a single image or folder of images
    """

    if torch.cuda.is_available()  :
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    model_path = os.path.join("../weight", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")  
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()



    # vs=cv2.VideoCapture('1.mp4')
    vs=cv2.VideoCapture(1)
    while vs.isOpened():
        success, input_image = vs.read()
        cv2.imshow('input_image', input_image)

        with torch.no_grad():
            original_width, original_height = input_image.shape[1],input_image.shape[0]
            input_image=transforms.ToTensor()(cv2.resize(input_image,(feed_width, feed_height))).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            cv2.imshow('frame',colormapped_im)

        if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    print('-> Done!')


if __name__ == '__main__':
    model_name='mono_640x192'


    test_simple(model_name)
