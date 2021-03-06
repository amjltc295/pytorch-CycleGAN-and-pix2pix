import os.path
import random

from PIL import Image
import torch

from data.base_dataset import BaseDataset, get_video_transform
from data.image_folder import make_video_dataset


class VideoDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_video_dirs = make_video_dataset(self.dir_A)
        self.B_video_dirs = make_video_dataset(self.dir_B)

        self.A_size = len(self.A_video_dirs)
        self.B_size = len(self.B_video_dirs)
        self.transform = get_video_transform(opt)

    def __getitem__(self, index):
        A_dir = self.A_video_dirs[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_dir = self.B_video_dirs[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))

        A_images = []
        B_images = []
        A_paths = []
        B_paths = []
        As = []
        Bs = []
        for i in range(self.opt.video_length):
            A_img = Image.open(A_dir[i*3]).convert('RGB')
            B_img = Image.open(B_dir[i*3]).convert('RGB')
            A_images.append(A_img)
            B_images.append(B_img)
            A_paths.append(A_dir[i])
            B_paths.append(B_dir[i])

        A_images = self.transform(A_images)
        B_images = self.transform(B_images)
        for i in range(self.opt.video_length):
            A = A_images[:, i, :, :]
            B = B_images[:, i, :, :]
            if self.opt.which_direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            if input_nc == 1:  # RGB to gray
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)
            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)
            As.append(A)
            Bs.append(B)

        return {'A': torch.stack(As, dim=1),
                'B': torch.stack(Bs, dim=1),
                'A_paths': A_paths, 'B_paths': B_paths}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'VideoDataset'
