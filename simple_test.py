import os
import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import argparse
from lib.utils import str2bool
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='dataset')
parser.add_argument('--dataset_root_dir', type=str, default='', help='dataset root dir')
parser.add_argument('--save_path', type=str, default='',  help='save results path')
parser.add_argument('--input_mask', type=str, default='pvnet',  help='save results path')
parser.add_argument('--visualization', type=str2bool, default=False,  help='visualization')
parser.add_argument('--gaussian_std', type=float, default=0.1,  help='gaussian_std')
parser.add_argument('--max_iteration', type=int, default=20,  help='max_iteration')
parser.add_argument('--tau', type=float, default=0.1,  help='tau')
parser.add_argument('--num_particles', type=int, default=180,  help='num_particles')
parser.add_argument('--w_o_CPN', type=str2bool, default=False,  help='with out CPN')
opt = parser.parse_args()

if opt.w_o_CPN == True:
    import lib.particle_filter_w_o_CPN as particle_filter
else:
    import lib.particle_filter_faster as particle_filter

if __name__ == '__main__':
    pf = particle_filter.ParticleFilter(opt.dataset, opt.dataset_root_dir, visualization=opt.visualization,
    gaussian_std=opt.gaussian_std, max_iteration=opt.max_iteration, tau=opt.tau, num_particles=opt.num_particles)

    img = np.load('sample/rgb.npy')
    depth = np.load('sample/depth.npy')
    label = np.load('sample/mask.npy')
    
    label = cv2.resize(label, (640,480), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)

    # cv2.imshow('label', label)
    # cv2.imshow('img', img)
    # cv2.imshow('depth', depth)
    # cv2.waitKey(0)
    label[label>0] = 1

    labels = label[label > 0]
    labels = np.unique(labels)

    objects_region = np.zeros((480,640))
    for labels_ in labels:
        label_region = ma.getmaskarray(ma.masked_equal(label, labels_))
        objects_region[label_region] = 1
        
    for itemid in labels:
        print('itemid: ', itemid)
        best_score, pose = pf.start(itemid, img, depth, label, objects_region, dataset=opt.dataset)
    # if best_score != 0:
    #     np.save(opt.save_path+pf.models[itemid-1]+"_"+str(now), pose)