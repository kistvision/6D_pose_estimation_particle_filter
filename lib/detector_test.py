import numpy as np
import yaml
import torch
import argparse
import os
from PIL import Image
import cv2
import pandas as pd
import json

from detector_models_cfg import create_model_detector

DATASET_ROOT = "/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/bop_datasets/"

class TensorCollection:
    def __init__(self, **kwargs):
        self.__dict__['_tensors'] = dict()
        for k, v in kwargs.items():
            self.register_tensor(k, v)

    def register_tensor(self, name, tensor):
        self._tensors[name] = tensor

    def delete_tensor(self, name):
        del self._tensors[name]

    def __repr__(self):
        s = self.__class__.__name__ + '(' '\n'
        for k, t in self._tensors.items():
            s += f'    {k}: {t.shape} {t.dtype} {t.device},\n'
        s += ')'
        return s

    def __getitem__(self, ids):
        tensors = dict()
        for k, v in self._tensors.items():
            tensors[k] = getattr(self, k)[ids]
        return TensorCollection(**tensors)

    def __getattr__(self, name):
        if name in self._tensors:
            return self._tensors[name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError

    @property
    def tensors(self):
        return self._tensors

    @property
    def device(self):
        return list(self.tensors.values())[0].device

    def __getstate__(self):
        return {'tensors': self.tensors}

    def __setstate__(self, state):
        self.__init__(**state['tensors'])
        return

    def __setattr__(self, name, value):
        if '_tensors' not in self.__dict__:
            raise ValueError('Please call __init__')
        if name in self._tensors:
            self._tensors[name] = value
        else:
            self.__dict__[name] = value

    def to(self, torch_attr):
        for k, v in self._tensors.items():
            self._tensors[k] = v.to(torch_attr)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def float(self):
        return self.to(torch.float)

    def double(self):
        return self.to(torch.double)

    def half(self):
        return self.to(torch.half)

    def clone(self):
        tensors = dict()
        for k, v in self.tensors.items():
            tensors[k] = getattr(self, k).clone()
        return TensorCollection(**tensors)


class PandasTensorCollection(TensorCollection):
    def __init__(self, infos, **tensors):
        super().__init__(**tensors)
        self.infos = infos.reset_index(drop=True)
        self.meta = dict()

    def register_buffer(self, k, v):
        assert len(v) == len(self)
        super().register_buffer()

    def merge_df(self, df, *args, **kwargs):
        infos = self.infos.merge(df, how='left', *args, **kwargs)
        assert len(infos) == len(self.infos)
        assert (infos.index == self.infos.index).all()
        return PandasTensorCollection(infos=infos, **self.tensors)

    def clone(self):
        tensors = super().clone().tensors
        return PandasTensorCollection(self.infos.copy(), **tensors)

    def __repr__(self):
        s = self.__class__.__name__ + '(' '\n'
        for k, t in self._tensors.items():
            s += f'    {k}: {t.shape} {t.dtype} {t.device},\n'
        s += f"{'-'*40}\n"
        s += '    infos:\n' + self.infos.__repr__() + '\n'
        s += ')'
        return s

    def __getitem__(self, ids):
        infos = self.infos.iloc[ids].reset_index(drop=True)
        tensors = super().__getitem__(ids).tensors
        return PandasTensorCollection(infos, **tensors)

    def __len__(self):
        return len(self.infos)

    def __getstate__(self):
        state = super().__getstate__()
        state['infos'] = self.infos
        state['meta'] = self.meta
        return state

    def __setstate__(self, state):
        self.__init__(state['infos'], **state['tensors'])
        self.meta = state['meta']
        return


class Detector:
    def __init__(self, model):
        model.eval()
        self.model = model
        self.config = model.config
        self.category_id_to_label = {v: k for k, v in self.config["label_to_category_id"].items()}

    def cast(self, obj):
        return obj.cuda()

    @torch.no_grad()
    def get_detections(self, images, detection_th=None,
                       output_masks=False, mask_th=0.8,
                       one_instance_per_class=False):
        images = self.cast(images).float()
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.max() > 1:
            images = images / 255.
            images = images.float().cuda()
        outputs_ = self.model([image_n for image_n in images])

        infos = []
        bboxes = []
        masks = []
        for n, outputs_n in enumerate(outputs_):
            outputs_n['labels'] = [self.category_id_to_label[category_id.item()] \
                                   for category_id in outputs_n['labels']]
            for obj_id in range(len(outputs_n['boxes'])):
                bbox = outputs_n['boxes'][obj_id]
                info = dict(
                    batch_im_id=n,
                    label=outputs_n['labels'][obj_id],
                    score=outputs_n['scores'][obj_id].item(),
                )
                mask = outputs_n['masks'][obj_id, 0] > mask_th
                bboxes.append(torch.as_tensor(bbox))
                masks.append(torch.as_tensor(mask))
                infos.append(info)

        if len(bboxes) > 0:
            bboxes = torch.stack(bboxes).cuda().float()
            masks = torch.stack(masks).cuda()
        else:
            infos = dict(score=[], label=[], batch_im_id=[])
            bboxes = torch.empty(0, 4).cuda().float()
            masks = torch.empty(0, images.shape[1], images.shape[2], dtype=torch.bool).cuda()

        outputs = PandasTensorCollection(
            infos=pd.DataFrame(infos),
            bboxes=bboxes,
        )
        if output_masks:
            outputs.register_tensor('masks', masks)
        if detection_th is not None:
            keep = np.where(outputs.infos['score'] > detection_th)[0]
            outputs = outputs[keep]

        if one_instance_per_class:
            infos = outputs.infos
            infos['det_idx'] = np.arange(len(infos))
            keep_ids = infos.sort_values('score', ascending=False).drop_duplicates('label')['det_idx'].values
            outputs = outputs[keep_ids]
            outputs.infos = outputs.infos.drop('det_idx', axis=1)
        return outputs

    
def load_detector(dataset):
    run_dir = DATASET_ROOT + 'checkpoints/detector-bop-{0}/'.format(dataset)
    with open(run_dir + 'config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        # cfg = check_update_config(cfg)
        label_to_category_id = cfg["label_to_category_id"]
        models = []
        for label in label_to_category_id:
            if label != "background":
                models.append(label)
        model = create_model_detector(cfg, len(label_to_category_id))
        ckpt = torch.load(run_dir + 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        model = Detector(model)
        return model, models

def load_dataset(dataset):
    path_list = {}
    if dataset == "tless":
        dir = DATASET_ROOT + '{0}/test_primesense/'.format(dataset)
    else:
        dir = DATASET_ROOT + '{0}/test/'.format(dataset)
    dir_lists = os.listdir(dir)
    dir_lists.sort()

    for folder in dir_lists:
        json_dir = dir + folder + "/scene_camera.json"
        with open(json_dir, "r") as json_file:
            camera_info = json.load(json_file)
            for scene in camera_info:
                img_path = dir + folder + '/rgb/{0}.png'.format('%06d' % int(scene))
                path_list[img_path] = [camera_info[scene]['cam_K'], camera_info[scene]['depth_scale']]
        #     # print(camera_info)
        # _dir = dir + folder + '/rgb/'
        # img_lists = os.listdir(_dir)
        # img_lists.sort()
        # for img in img_lists:
        #     path_list.append(_dir + img)
    return path_list

# def init_detector(dataset):
#     datasets = load_dataset(dataset)
#     model = load_detector(dataset)


#     for rgb_path in datasets:
#         rgb_cv2 = cv2.imread(rgb_path)
#         print("*** {0} ***".format(rgb_path))
#         rgb = np.array(Image.open(rgb_path))
#         if rgb.ndim == 2:
#             rgb = np.repeat(rgb[..., None], 3, axis=-1)
#         rgb = rgb[..., :3]
#         h, w, c = rgb.shape
#         rgb = torch.as_tensor(rgb)#.cuda()
#         # rgb = rgb.view(c, h, w).contiguous()
#         rgb = rgb.unsqueeze(0).contiguous()
#         # rgb = rgb.cuda().float().permute(0, 3, 1, 2) / 255

#         output = model.get_detections(rgb, one_instance_per_class=True, detection_th=0.7, output_masks=True)
#         print(output)
#         masks = output.masks.cpu().data.numpy()
#         labels = output.infos.label#.cpu().data.numpy()
#         bboxes = output.bboxes.cpu().data.numpy()
#         for i in range(len(masks)):
#             cv2.imshow("{0}".format(labels[i]), masks[i] * 1.0)
#         cv2.imshow("rgb", rgb_cv2)
#         cv2.waitKey(0)


def main():
    datasets = load_dataset('ycbv')
    model = load_detector('ycbv')

    for rgb_path in datasets:
        rgb_cv2 = cv2.imread(rgb_path)
        print("*** {0} ***".format(rgb_path))
        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = rgb[..., :3]
        h, w, c = rgb.shape
        rgb = torch.as_tensor(rgb)#.cuda()
        # rgb = rgb.view(c, h, w).contiguous()
        rgb = rgb.unsqueeze(0).contiguous()
        # rgb = rgb.cuda().float().permute(0, 3, 1, 2) / 255

        output = model.get_detections(rgb, one_instance_per_class=True, detection_th=0.7, output_masks=True)
        print(output)
        masks = output.masks.cpu().data.numpy()
        labels = output.infos.label#.cpu().data.numpy()
        bboxes = output.bboxes.cpu().data.numpy()
        for i in range(len(masks)):
            cv2.imshow("{0}".format(labels[i]), masks[i] * 1.0)
        cv2.imshow("rgb", rgb_cv2)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
