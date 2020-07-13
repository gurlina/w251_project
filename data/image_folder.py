###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

#LG
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

#LG 
VIDEO_EXTENSIONS = [
    '.mp4', '.MP4', '.avi', '.AVI',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


#LG
def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)

#LG
def get_video_file_name(dir):
    vfn_path = ''
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_video_file(fname):
                vfn_path = os.path.join(root, fname)
                break

    return vfn_path

#LG
def get_frames(image_dir):
    frames = []
    all_path = []
    filename = get_video_file_name(image_dir)

    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #v_cap = cv2.VideoCapture(0)
    #v_len = 300
    print("Number of frames in video: ", v_len)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            print("Failed to retrieve frame: ", fn)
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        path = os.path.join(image_dir, "frame"+str(fn)+".jpg")
        frames.append(frame)
        all_path.append(path)
    v_cap.release()
    return frames, all_path


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def store_dataset(dir):
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path).convert('RGB')
                images.append(img)
                all_path.append(path)

    return images, all_path


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
