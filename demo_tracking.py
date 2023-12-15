import os
import sys
import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import argparse
from lib.utils import str2bool
from tqdm import tqdm
import time
import yolact.load_model as yolact_impl
import torch
from yolact.utils.augmentations import FastBaseTransform
from threading import Thread

opt = yolact_impl.get_opt()


import lib.particle_filter_tracking as particle_filter
import pyrealsense2 as rs

import zmq

if __name__ == '__main__':

    # context = zmq.Context()
    # socket = context.socket(zmq.PUSH)
    # # socket = context.socket(zmq.REP)
    # socket.bind("tcp://*:5555")

    #  Socket to talk to server
    # print("Connecting to hello world server…")

    yolact = yolact_impl.get_yolact_model()
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    cx = color_intrin.ppx
    cy = color_intrin.ppy
    fx = color_intrin.fx
    fy = color_intrin.fy

    init_iters = 30
    pf = particle_filter.ParticleFilter(cx, cy, fx, fy, opt.dataset_pf, visualization=opt.visualization,
    gaussian_std=opt.gaussian_std, max_iteration=init_iters, tau=opt.tau, num_particles=opt.num_particles)

    sending_poses = np.zeros((4, 7))
    sending_poses[:,2] = 1000.0

    # class myClassA(Thread):
    #     def __init__(self):
    #         Thread.__init__(self)
    #         self.daemon = True
    #         self.start()
    #     def run(self):
    #         while True:
    #             # message = socket.recv()
    #             # socket.connect("tcp://localhost:5555")
    #             socket.send(bytes(sending_poses.tobytes()))  # pose = [itemid, x, y, z, q0, q1, q2, q3]

    # myClassA()

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # r, g, b = cv2.split(color_image)
        # color_image = cv2.merge((b, g, r))

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

        try:
            frame = torch.from_numpy(np.array(color_image)).cuda().float()
        except:
            continue
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = yolact(batch)
        try:
            masks, classes, boxes, img_numpy = yolact_impl.prep_display(preds, frame, None, None, undo_transform=False)
        except:
            continue
        # return 0, 0, 0, rgb_input

        img_numpy = np.array(img_numpy)[:, :, :3]
        img_numpy.astype(np.float32)

        # print(masks.shape)
        # print(classes.shape)
        # print(boxes.shape)

        # print(masks.shape)
        # Initialize sending poses (default position: [0,0,1000.0])
        # sending_poses = np.zeros((21, 7))
        # sending_poses = np.zeros((4, 7))
        # sending_poses[:,2] = 1000.0
        # print("default sending_poses")
        # print(sending_poses)
        detected_objects = [False for _ in range(4)]

        cv2.imshow("yolact detection", img_numpy)

        items = []

        objects_region = np.zeros((480,640))
        for i in range(len(masks)):
            label = masks[i, :, :, 0]
            objects_region[label>0] = 1
        # cv2.imshow("objects_region", objects_region)
        # print("====================")

        for index in range(len(masks)):
            itemid = classes[index] + 1
            if itemid not in items:
                items.append(itemid)
            else:
                continue
            box = boxes[index]
            label = masks[index, :, :, 0]
            best_score, pose, color_image = pf.start(itemid, color_image, aligned_depth_image, label, objects_region, box)
            if best_score == 0:
                continue

            # map itemid 3 -> 0  /  4 -> 1  /  5 -> 2  /  12 -> 3
            if itemid == 3:
                itemid = 0
                detected_objects[0] = True
            elif itemid == 4:
                itemid = 1
                detected_objects[1] = True
            elif itemid == 5:
                itemid = 2
                detected_objects[2] = True
            elif itemid == 12:
                itemid = 3
                detected_objects[3] = True
            else:
                continue
            sending_poses[itemid][:] = pose
        # Send itemid, pose to the server
        # while True:
        #     try:
        #         # print("sending_poses")
        #         message = socket.recv()
        #         # print(message)
        #         # socket.connect("tcp://192.168.50.13:5555")
        #         socket.connect("tcp://localhost:5555")
        #         socket.send(bytes(sending_poses.tobytes()))  # pose = [itemid, x, y, z, q0, q1, q2, q3]
        #         break
        #     except:
        #         pass

            #  Get the reply.
            # message = socket.recv()
        for itemid in range(len(detected_objects)):
            if not detected_objects[itemid]:
                sending_poses[itemid,:] = 0
                sending_poses[itemid,2] = 1000.0

        cv2.imshow("estimated 6d objects pose", color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('1'):
            pf.initialized["004_sugar_box"] = False
        elif key == ord('2'):
            pf.initialized["005_tomato_soup_can"] = False
        elif key == ord('3'):
            pf.initialized["006_mustard_bottle"] = False
        elif key == ord('4'):
            pf.initialized["021_bleach_cleanser"] = False