
import os
import sys
import cv2
import time

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath)
sys.path.append(libpath + '/../CenterFindNet/lib')
import render
import objloader
# import ctypes
from PIL import Image
import numpy as np
import scipy.io as scio
import random
import time
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import matplotlib.pyplot as plt
import numpy.ma as ma
import open3d as o3d

from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample

from sample import *
from .utils import quaternion_matrix, calc_pts_diameter, draw_object
from scipy import spatial

import torch
from torch.autograd import Variable
from centroid_prediction_network import CentroidPredictionNetwork

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class ParticleFilter():
    def __init__(self, cx, cy, fx, fy, dataset="ycb", visualization=False, gaussian_std=0.1, max_iteration=20, tau=0.1, num_particles=180, w_o_CPN=False, w_o_Scene_occlusion=False):
        self.cam_cx = cx
        self.cam_cy = cy
        self.cam_fx = fx
        self.cam_fy = fy
        self.cam_scale = 0.001

        # rotation / translation propagation constant
        self.Cr = 0.5
        self.Ct = 0.5
        if dataset == "ycb":
            self.ycb_toolbox_dir = libpath + '/../CenterFindNet/YCB_Video_toolbox'
            self.dataset_config_dir = libpath + '/../CenterFindNet/datasets/dataset_config'
            self.cad_model_root_dir = libpath + '/../models/ycb/'
            self.w_o_CPN = w_o_CPN
            self.w_o_Scene_occlusion = w_o_Scene_occlusion

            self.particle_filter_info_dict = {}
            self.particle_filter_info = ""

            self.models = [
                    "002_master_chef_can",      # 1
                    "003_cracker_box",          # 2
                    "004_sugar_box",            # 3
                    "005_tomato_soup_can",      # 4
                    "006_mustard_bottle",       # 5
                    "007_tuna_fish_can",        # 6
                    "008_pudding_box",          # 7
                    "009_gelatin_box",          # 8
                    "010_potted_meat_can",      # 9
                    "011_banana",               # 10
                    "019_pitcher_base",         # 11
                    "021_bleach_cleanser",      # 12
                    "024_bowl",                 # 13
                    "025_mug",                  # 14
                    "035_power_drill",          # 15
                    "036_wood_block",           # 16
                    "037_scissors",             # 17
                    "040_large_marker",         # 18
                    "051_large_clamp",          # 19
                    "052_extra_large_clamp",    # 20
                    "061_foam_brick"            # 21
            ]

            # Same number of particles (180) for all objects.
            self.num_particles = [num_particles * 2, num_particles * 2, num_particles, num_particles * 2, num_particles* 2, num_particles*2,
            num_particles, num_particles*2, num_particles*2, num_particles, num_particles*2, num_particles*2, num_particles,
            num_particles, int(num_particles/2), num_particles*2, int(num_particles/2), num_particles, int(num_particles/2), int(num_particles/2), num_particles*2]

            # self.taus = [tau, tau * 2, tau, tau, tau, tau, tau, tau, tau, tau,
            #             tau, tau, tau, tau, tau * 2, tau, tau * 2, tau, tau * 2, tau * 2, tau]
            # self.taus = [tau, tau, tau, tau, tau, tau * 0.5, tau, tau, tau, tau,
            #             tau, tau, tau, tau, tau, tau, tau , tau, tau , tau, tau]
            # self.tau = tau
            # self.taus = [tau, tau * 1.5, tau, tau, tau, tau, tau, tau, tau, tau,
            #             tau, tau, tau * 0.5, tau, tau * 1.5, tau, tau * 1.5, tau, tau * 2.0, tau * 2.0, tau]
            self.taus = [0.1, 0.15, 0.1, 0.05, 0.1,
                         0.1, 0.1, 0.1, 0.1, 0.1, 
                         0.1, 0.1, 0.05, 0.1, 0.2, 
                         0.1, 0.2, 0.1, 0.2, 0.2, 0.1]

            self.testlist = []
            input_file = open('{0}/test_data_list.txt'.format(self.dataset_config_dir))
            # input_file = open('{0}/small_test_data_list.txt'.format(self.dataset_config_dir))
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.testlist.append(input_line)
            input_file.close()
        
        elif dataset == "lmo":
            self.cam_cx = 325.2611
            self.cam_cy = 242.04899
            self.cam_fx = 572.4114
            self.cam_fy = 573.57043
            self.cam_scale = 0.001
            self.cad_model_root_dir = libpath + '/../models/lmo/'
            # self.models = {1:'Ape', 5:'Can', 6:'Cat', 8:'Driller', 9:'Duck', 10:'Eggbox', 11:'Glue', 12:'Holepuncher'}
            self.models = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
            self.models_id = [1, 5, 6, 8, 9, 10, 11, 12]
            # Same number of particles (720) for all objects.
            self.num_particles = [num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2]
            self.taus = [tau * 2, tau * 2, tau * 2, tau * 2, tau * 2, tau * 2, tau * 2, tau * 2]
        else:
            print("Write your own dataset config here.")
            exit(0)

        if self.w_o_CPN:
            self.info = {'Height':480, 'Width':640, 'fx':self.cam_fx, 'fy':self.cam_fy, 'cx':self.cam_cx, 'cy':self.cam_cy, 'num_particles':num_particles*5}
        else:
            self.info = {'Height':480, 'Width':640, 'fx':self.cam_fx, 'fy':self.cam_fy, 'cx':self.cam_cx, 'cy':self.cam_cy, 'num_particles':num_particles}
        render.setup(self.info)

        self.visualization = visualization
        self.num_points = 1000
        # self.gaussian_std = gaussian_std
        self.gaussian_std = [0.2, 0.05, 0.1, 0.2, 0.1,
                             0.2, 0.2, 0.2, 0.2, 0.1,
                             0.15, 0.1, 0.1, 0.15, 0.1,
                             0.05, 0.05, 0.2, 0.05, 0.05, 0.05]
        # self.gaussian_std = [0.1, 0.1, 0.1, 0.3, 0.2, 0.2, 0.1, 0.3, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
        # self.max_iteration = max_iteration

        self.min_particles = 256
        
        self.K = [[self.cam_fx, 0, self.cam_cx],
            [0, self.cam_fy, self.cam_cy],
            [0, 0, 1]]
        self.K = np.array(self.K)

              
        """ Load Centroid Prediction Network """
        model_path = libpath + '/../CenterFindNet/trained_model/CPN_model_91_0.00023821471899932882.pth'
        # model_path = os.path.dirname(os.path.abspath(__file__)) +
        self.estimator = CentroidPredictionNetwork(num_points = self.num_points)
        self.estimator.load_state_dict(torch.load(model_path))
        self.estimator.eval()
        self.estimator.cuda()

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.context = {}
        self.diameters = {}
        self.model_points = {}
        self.np_model_points = {}
        self.corners = {}
        self.prev_values = {}
        self.initialized = {}
        self.max_iteration = {}
        for model in self.models:
            print("*** " + model + " adding... ***")
            self.prev_values[model] = {}
            self.initialized[model] = False
            self.max_iteration[model] = max_iteration

            self.prev_values[model]['poses'] = None
            self.prev_values[model]['pose_distribution'] = None
            self.prev_values[model]['weights'] = None
            self.prev_values[model]['centroid'] = None
            self.prev_values[model]['scores'] = None
            self.prev_values[model]['best_score'] = 0
            self.prev_values[model]['final_pose'] = None

            cad_model_path = self.cad_model_root_dir + '{}/textured_simple.obj'.format(model)
            pcd_path = self.cad_model_root_dir + '{}/textured_simple.pcd'.format(model)

            V, F = objloader.LoadTextureOBJ_VF_only(cad_model_path)
            self.context[model] = render.SetMesh(V, F)

            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = np.asarray(pcd.points)

            x_max = np.max(pcd[:,0])
            x_min = np.min(pcd[:,0])
            y_max = np.max(pcd[:,1])
            y_min = np.min(pcd[:,1])
            z_max = np.max(pcd[:,2])
            z_min = np.min(pcd[:,2])

            corner = np.array([[x_min, y_min, z_min],
                                        [x_max, y_min, z_min],
                                        [x_max, y_max, z_min],
                                        [x_min, y_max, z_min],

                                        [x_min, y_min, z_max],
                                        [x_max, y_min, z_max],
                                        [x_max, y_max, z_max],
                                        [x_min, y_max, z_max]])
            self.corners[model] = corner

            self.diameters[model] = calc_pts_diameter(pcd)

            # print("self.diameters[model] ; ", self.diameters[model] )

            dellist = [j for j in range(0, len(pcd))]
            dellist = random.sample(dellist, len(pcd) - self.num_points)

            model_point = np.delete(pcd, dellist, axis=0)
            np_model_point = model_point.copy()

            model_point = torch.from_numpy(model_point.astype(np.float32)).view(1, self.num_points, 3)
            self.model_points[model] = model_point.cuda()
            self.np_model_points[model] = np_model_point

        self.rotation_samples = {}
        for i, model in enumerate(self.models):
            self.rotation_samples[model] = get_rotation_samples_perch(model, num_samples=self.num_particles[i])
            print(len(self.rotation_samples[model]))

        self.particle_filter_info_dict["num_particles"] = self.num_particles
        self.particle_filter_info_dict["taus"] = self.taus
        self.particle_filter_info_dict["gaussian_std"] = self.gaussian_std
        self.particle_filter_info_dict["max_iteration"] = self.max_iteration

        self.particle_filter_info = str(self.particle_filter_info_dict)
        # print(self.particle_filter_info)
        # exit(0)


    def mat2pdf_np(self, distance_matrix, mean, std):
        coeff = 1/(np.sqrt(2*np.pi) * std)
        pdf = coeff * np.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def mat2pdf(self, distance_matrix, mean, std):
        coeff = torch.ones_like(distance_matrix) * (1/(np.sqrt(2*np.pi) * std))
        mean = torch.ones_like(distance_matrix) * mean
        std = torch.ones_like(distance_matrix) * std
        pdf = coeff * torch.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def estimate(self, pos, weights):
        """returns mean and variance of the weighted particles"""
        mean = np.average(pos, weights=weights, axis=0)
        var  = np.average((pos - mean)**2, weights=weights, axis=0)
        return mean, var

    def start(self, itemid, img, depth, label, objects_region, rois=[], target_t=None):
        model = self.models[itemid-1]
        prev_poses = self.prev_values[model]['poses']
        prev_pose_distribution = self.prev_values[model]['pose_distribution']
        prev_weights = self.prev_values[model]['weights']
        prev_centroid = self.prev_values[model]['centroid']
        prev_scores = self.prev_values[model]['scores']
        context = self.context[model]
        diameter = self.diameters[model]
        model_point = self.model_points[model]
        np_model_point = self.np_model_points[model]
        corner = self.corners[model]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        # mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
        mask = label * mask_depth
        masked_depth = mask * depth * self.cam_scale

        rmin = int(rois[1])
        rmax = int(rois[3])
        cmin = int(rois[0])
        cmax = int(rois[2])

        masked_depth_copy = masked_depth.copy()
        masked_depth_copy = masked_depth_copy[masked_depth_copy > 0]

        var = np.var(masked_depth_copy)
        mean = np.mean(masked_depth_copy)
        masked_depth[masked_depth -mean > var + diameter] = 0

        if self.w_o_Scene_occlusion:
            other_objects_region = np.zeros((480, 640))
        else:
            other_objects_region = objects_region.copy()
            other_objects_region[label>0] = 0
            depth_zero_in_mask = np.logical_and(label, np.logical_not(depth))
            other_objects_region[depth_zero_in_mask] = 1

        # cv2.imshow("other_objects_region", other_objects_region)
        # cv2.waitKey(0)
        """ Initial pose hypotheses """
        poses = []
        final_pose = self.prev_values[model]['final_pose']
        best_score = self.prev_values[model]['best_score']

        # pose_distribution = None

        """ With CPN"""
        choose = masked_depth[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            if len(choose) == 0:
                return 0, 0, 0
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')
            # return 0, 0

        depth_masked = masked_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        pt2 = depth_masked #* cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        
        while not torch.cuda.is_available():
            time.sleep(0.001)

        cloud = torch.from_numpy(cloud.astype(np.float32)).view(1, self.num_points, 3).cuda()

        """ Centroid prediction for generating the initial translation of the particle filter system. """
        if target_t is not None:
            centroid = target_t
        else:
            # print(cloud)
            # print(model_point)
            # print(type(cloud))
            # print(type(model_point))
            centroid = self.estimator(cloud, model_point)
            centroid = centroid[0,0,:].cpu().data.numpy()

        if not self.initialized[model]:
            for sample_ryp in self.rotation_samples[model]:
                quat = list(euler2quat(sample_ryp[0], sample_ryp[1], sample_ryp[2]))
                pose = np.hstack([centroid[0], centroid[1], centroid[2], quat])
                poses.append(pose)

            prev_poses = None
            prev_pose_distribution = None
            prev_weights = None
            prev_centroid = None
            prev_scores = None
            self.max_iteration[model] = 50
            self.Cr = 0.5

        render.setSrcDepthImage(self.info, masked_depth.copy(), other_objects_region.copy())


        for iters in range(self.max_iteration[model]):
            # print(self.max_iteration[model])
            if not (prev_poses is None and prev_pose_distribution is None and prev_weights is None and prev_centroid is None and prev_scores is None):
                """ Resampling"""
                N = len(prev_weights)
                positions = (np.random.random() + np.arange(N)) / N
                indexes = np.zeros(N, "i")
                cumulative_sum = np.cumsum(prev_weights)
                i, j = 0, 0
                while i < N and j < N:
                    if positions[i] < cumulative_sum[j]:
                        indexes[i] = j
                        i += 1
                    else:
                        j += 1
                values, counts = np.unique(indexes, return_counts=True)

                poses = []
                pose_distribution = []
                delta_centroid = (centroid - prev_centroid) 

                for count_i, index in enumerate(values):
                    count = counts[count_i]
                    pose_copy = prev_poses[index]
                    score = prev_scores[index]
                    count = int(count)

                    mu, var = pose_copy[0], 0.01 * pow(1-score, 2)
                    t1 = np.random.normal(mu, var, count)
                    mu, var = pose_copy[1], 0.01 * pow(1-score, 2)
                    t2 = np.random.normal(mu, var, count)
                    mu, var = pose_copy[2], 0.01 * pow(1-score, 2)
                    t3 = np.random.normal(mu, var, count)

                    trans = np.asarray([t1, t2, t3])
                    trans += np.tile(delta_centroid, reps=[count,1]).T

                    mu, var = pose_copy[3], (np.pi * self.Cr) * pow(1-score, 2)
                    q0 = np.random.normal(mu, var, count)
                    mu, var = pose_copy[4], (np.pi * self.Cr) * pow(1-score, 2)
                    q1 = np.random.normal(mu, var, count)
                    mu, var = pose_copy[5], (np.pi * self.Cr) * pow(1-score, 2)
                    q2 = np.random.normal(mu, var, count)
                    mu, var = pose_copy[6], (np.pi * self.Cr) * pow(1-score, 2)
                    q3 = np.random.normal(mu, var, count)

                    quat = np.asarray([q0, q1, q2, q3])

                    for q in quat:
                        for i in range(len(q)):
                            if q[i] < -1:
                                q[i] = np.trunc(q[i])-1 - q[i]
                            if q[i] > 1:
                                q[i] = np.trunc(q[i])+1 - q[i]

                    pose = np.vstack((trans, quat)).T
                    poses.extend(pose)

                    pose_dist_copy = [prev_pose_distribution[index] for i in range(count)]
                    pose_distribution.extend(pose_dist_copy)

            # if iters < 5:
            #     threshold = self.taus[itemid-1] / (iters * 2+1)
            # else:
            #     threshold = self.taus[itemid-1] / 10.0
            threshold = self.taus[itemid-1] / 2.0


            render.setNumOfParticles(len(poses), int(threshold * 1000000))
            transform_matrixes = np.zeros((len(poses), 4, 4))
            for i in range(len(poses)):
                pose = poses[i]

                transform_matrix = quaternion_matrix(pose[3:]).astype(np.float32)
                transform_matrix[:3,3] = pose[:3]
                transform_matrixes[i] = transform_matrix
            transform_matrixes = transform_matrixes.astype(np.float32)
            render.render(context, transform_matrixes)

            # rendered_image = render.getDepth(self.info)
            # for im in rendered_image:
            #     cv2.imshow("rendered_image", im)
            #     cv2.waitKey(0)

            """ Access to the CUDA memory to get the scores of each pose hypothesis. """
            scores = render.getMatchingScores(len(poses))

            # exit(0)

            if len(scores[scores > 0]) == 0:
                continue
                # return 0, 0

            """ Likelihood calculation """
            pdf_matrix = self.mat2pdf_np(scores / max(scores), 1, self.gaussian_std[itemid-1])
            # pdf_matrix = self.mat2pdf_np(scores / max(scores), 1, self.gaussian_std)
            if iters == 0:
                pose_distribution = pdf_matrix / np.sum(pdf_matrix + 1e-8)
                weights = pose_distribution
            else:
                pose_distribution = np.exp(np.log(pdf_matrix + 1e-8) + np.log(pose_distribution))
                weights = pose_distribution / np.sum(pose_distribution + 1e-8)
                pose_distribution = weights



            """ Current state estimation by weighted average. """
            mu, var = self.estimate(poses, weights)

            render.setNumOfParticles(1, int(threshold * 1000000))
            transform_matrix = quaternion_matrix(mu[3:]).astype(np.float32)
            transform_matrix[:3,3] = mu[:3]
            render.render(context, transform_matrix)
            matching_score = render.getMatchingScores(1)

            # print("="*50)
            # print("Current iteration : ", iters)
            # print("Best score : ", best_score)
            # print("Best matching score : ", max(scores))
            # print("particles size : ", len(poses))
            # print("best pose : ", final_pose)
            # print("Rendering and calc score time: ", time.time() - Rendering_time)

            if matching_score > best_score:
                best_score = matching_score
                final_pose = mu
                # Taking pose of the max score method. This has lower performance than the weighted average method.
                # final_pose = poses[scores.argmax()]


            """ 
                Visualization
            """
            if self.visualization == True and final_pose is not None:
                transform_matrix = quaternion_matrix(final_pose[3:]).astype(np.float32)
                transform_matrix[:3,3] = final_pose[:3]
                final_pose[3:] = mat2quat(transform_matrix[:3,:3])

                # draw_box = img.copy()
                try:
                    draw_object(itemid, self.K, transform_matrix[:3,:3], transform_matrix[:3,3], img, np_model_point, corner)
                    # cv2.imshow("draw_box", draw_box)
                    # key = cv2.waitKey(1)
                    # if key == ord('q'):
                    #     # exit(0)
                    #     return 0, 0
                except:
                    pass
                # cv2.imwrite("/home/user/python_projects/6D_pose_estimation_particle_filter/results/images/{0}_{1}.png".format(model, str(iters)), draw_box)



            """ Propagation """
            prev_poses = poses.copy()
            prev_pose_distribution = pose_distribution.copy()
            prev_weights = weights.copy()
            prev_centroid = centroid.copy()
            prev_scores = scores.copy()
            self.prev_values[model]['poses'] = prev_poses
            self.prev_values[model]['pose_distribution'] = prev_pose_distribution
            self.prev_values[model]['weights'] = prev_weights
            self.prev_values[model]['centroid'] = prev_centroid
            self.prev_values[model]['scores'] = prev_scores

        if not self.initialized[model]:
            self.initialized[model] = True
            self.max_iteration[model] = 4
            self.Cr = 0.25
            # self.Ct = 0.25
        pose = final_pose
        # self.prev_values[model]['best_score'] = best_score
        # self.prev_values[model]['final_pose'] = final_pose
        return best_score, pose, img
