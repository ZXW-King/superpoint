# -*-coding:utf8-*-
import os
import glob
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.params import dict_update
from dataset.utils.homographic_augmentation import homographic_aug_pipline
from dataset.utils.photometric_augmentation import PhotoAugmentor
from utils.keypoint_op import compute_keypoint_map_xy
from dataset.utils.photometric_augmentation import *
import cv2


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, config, is_train, device='cpu'):

        super(COCODataset, self).__init__()
        self.device = device
        self.is_train = is_train
        self.resize = tuple(config['resize'])
        self.photo_augmentor = PhotoAugmentor(config['augmentation']['photometric'])
        # load config
        self.config = config  # dict_update(getattr(self, 'default_config', {}), config)
        # get images
        if self.is_train:
            self.samples = self._init_data(config['image_train_path'], config['label_train_path'])
        else:
            self.samples = self._init_data(config['image_test_path'], config['label_test_path'])

    def _init_data(self, image_path, label_path=None):
        ##
        if not isinstance(image_path, list):
            image_paths, label_paths = [image_path, ], [label_path, ]
        else:
            image_paths, label_paths = image_path, label_path

        samples = []
        for im_path, lb_path in zip(image_paths, label_paths):
            if lb_path == '':
                continue

            self.flag = None
            if os.path.isdir(lb_path):
                csv_datas = os.listdir(lb_path)
                temp = []
                for file in csv_datas:
                    csv_path = os.path.join(lb_path,file)
                    temp.append((im_path,csv_path))
                self.flag = "dir"
            elif os.path.isfile(lb_path):
                self.flag = "file"
                with open(lb_path, "r") as f:
                    data = f.readlines()
                    if lb_path.endswith('csv'):
                        data = data[1:]

                data = [d.strip().split(',') for d in data]
                temp = [
                    {'image': os.path.join(im_path, d[0].strip()),
                     'label': np.array(d[1:]).astype('float').reshape(-1, 3)}
                    for d in data]
            else:
                temp = []

            samples += temp
        ##
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''load raw data'''
        data_path = self.samples[idx]  # raw image path of processed image and point path
        if self.flag == "file":
            img = cv2.imread(data_path['image'], 0)  # Gray image
            pts = None if data_path['label'] is None else data_path['label'][:, 0:2]  # N*2,xy
        elif self.flag == "dir":
            with open(data_path[1], "r") as f:
                datas = f.readlines()
                filename = os.path.basename(data_path[1])
                if filename.endswith('csv'):
                    img_name = datas[0].split(",")[0].strip()
                    image_file = os.path.join(data_path[0], img_name)
                    img = cv2.imread(image_file, 0)  # Gray image
                    landmarks = np.array(datas[0].split(",")[1:])
                    landmarks = landmarks.astype('float').reshape(-1, 3)  # 转为1行3列  （x,y,p）
                    pts = None if landmarks is None else landmarks[:, 0:2]  # N*2,xy
        else:
            raise Exception("no data!")
        h, w = img.shape
        kpts_tensor, kpts_map = None, None

        if pts is not None:
            id = np.argwhere(pts[:, 0] < w)
            pts = pts[id].squeeze(axis=1)
            id = np.argwhere(pts[:, 1] < h)
            pts = pts[id].squeeze(axis=1)

            img = cv2.resize(img, self.resize[::-1])


            # resize pts
            H, W = self.resize
            pts[:, 0] = pts[:, 0] * W / w
            pts[:, 1] = pts[:, 1] * H / h

            # init data dict
            img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=self.device)
            kpts_tensor = torch.as_tensor(pts, dtype=torch.float, device=self.device)
            kpts_map =  compute_keypoint_map_xy(kpts_tensor, img.shape, device=self.device)

        valid_mask = torch.ones(img.shape, dtype=torch.float, device=self.device)

        data = {'raw': {'img': img_tensor,
                        'kpts': kpts_tensor,
                        'kpts_map': kpts_map,
                        'mask': valid_mask},
                'warp': None,
                'homography': torch.eye(3, device=self.device)}
        data['warp'] = deepcopy(data['raw'])

        ##
        if self.is_train:
            photo_enable = self.config['augmentation']['photometric']['train_enable']
            homo_enable = self.config['augmentation']['homographic']['train_enable']
        else:
            photo_enable = self.config['augmentation']['photometric']['test_enable']
            homo_enable = self.config['augmentation']['homographic']['test_enable']

        if homo_enable and data['raw']['kpts'] is not None:  # homographic augmentation
            # return dict{warp:{img:[H,W], point:[N,2], valid_mask:[H,W], homography: [3,3]; tensors}}
            data_homo = homographic_aug_pipline(data['warp']['img'],
                                                data['warp']['kpts'],
                                                self.config['augmentation']['homographic'],
                                                device=self.device, transpose=True)
            data.update(data_homo)

        if photo_enable:
            photo_img = data['warp']['img'].cpu().numpy().round().astype(np.uint8)
            photo_img = self.photo_augmentor(photo_img)
            data['warp']['img'] = torch.as_tensor(photo_img, dtype=torch.float, device=self.device)

        ##normalize
        data['raw']['img'] = data['raw']['img'] / 255.
        data['warp']['img'] = data['warp']['img'] / 255.

        return data  # img:HW, kpts:N2, kpts_map:HW, valid_mask:HW, homography:HW

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `img`, `img_name`, `kpts`, `kpts_map`,
        `valid_mask`, `homography`...
        img:H*W, kpts:N*2, kpts_map:HW, valid_mask:HW, homography:HW
        :return:
        """
        sub_data = {'img': [], 'kpts_map': [], 'mask': []}  # remove kpts
        batch = {'raw': sub_data, 'warp': deepcopy(sub_data), 'homography': []}
        for s in samples:
            batch['homography'].append(s['homography'])
            # batch['img_name'].append(s['img_name'])
            for k in sub_data:
                if k == 'img':
                    batch['raw'][k].append(s['raw'][k].unsqueeze(dim=0))
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k].unsqueeze(dim=0))
                else:
                    batch['raw'][k].append(s['raw'][k])
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k])
        ##
        batch['homography'] = torch.stack(batch['homography'])
        for k0 in ('raw', 'warp'):
            for k1 in sub_data:  # `img`, `img_name`, `kpts`, `kpts_map`...
                if k1 == 'kpts' or k1 == 'img_name':
                    continue
                batch[k0][k1] = torch.stack(batch[k0][k1])

        return batch


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    from dataset.utils.photometric_augmentation import *

    with open('../config/superpoint_train.yaml', 'r') as fin:
        config = yaml.load(fin)

    coco = COCODataset(config['data'], True)
    cdataloader = DataLoader(coco, collate_fn=coco.batch_collator, batch_size=1, shuffle=True)

    for i, d in enumerate(cdataloader):
        if i >= 10:
            break
        img = (d['raw']['img'] * 255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        img_warp = (d['warp']['img'] * 255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        img = cv2.merge([img, img, img])
        img_warp = cv2.merge([img_warp, img_warp, img_warp])
        ##
        kpts = np.where(d['raw']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img, (kp[1], kp[0]), radius=3, color=(0, 255, 0))
        kpts = np.where(d['warp']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img_warp, (kp[1], kp[0]), radius=3, color=(0, 255, 0))

        mask = d['raw']['mask'].cpu().numpy().squeeze().astype(np.int).astype(np.uint8) * 255
        warp_mask = d['warp']['mask'].cpu().numpy().squeeze().astype(np.int).astype(np.uint8) * 255

        img = cv2.resize(img, (640, 480))
        img_warp = cv2.resize(img_warp, (640, 480))

        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        plt.imshow(mask)
        plt.subplot(2, 2, 3)
        plt.imshow(img_warp)
        plt.subplot(2, 2, 4)
        plt.imshow(warp_mask)
        plt.show()

    print('Done')
