import numpy as np
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.datasets import DatasetFolder
import os
import PIL.Image as Image


class ImageFolder():
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_size=224, weight_type=None, gamma=1.5, base_transform=None,
                 crop_transform=None, flip_transform=None, ending_transform=None):
        self.root = root
        self.data = [os.path.join(self.root, img) for img in os.listdir(root)]
        self.img_size = img_size
        self.weight_type = weight_type
        self.gamma = gamma
        self.base_transform = base_transform
        self.crop_transform = crop_transform
        self.flip_transform = flip_transform
        self.ending_transform = ending_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = Image.open(self.data[index]).convert("RGB")

        # base transform (might be empty)
        if self.base_transform is not None:
            q, k = self.base_transform(sample), self.base_transform(sample)
        else:
            q, k = sample, sample

        q, is_flip = self.flip_transform(q)
        k, _ = self.flip_transform(k, is_flip=is_flip)

        q, (t_q, l_q, h_q, w_q) = self.crop_transform(q)
        k, (t_k, l_k, h_k, w_k) = self.crop_transform(k)

        b_q, r_q = t_q + h_q, l_q + w_q
        b_k, r_k = t_k + h_k, l_k + w_k

        label_weight = 1
        if self.weight_type == 'iou':
            label_weight = (self.iou((t_q, l_q, b_q, r_q), (t_k, l_k, b_k, r_k))) ** self.gamma
        if self.weight_type == 'ioa':
            label_weight = (self.ioa((t_q, l_q, b_q, r_q), (t_k, l_k, b_k, r_k))) ** self.gamma
        label_weight = np.array(label_weight, dtype=np.float)

        # === C4 ===
        anchor_sizes = [48]
        stride = 32.
        anchors_q, anchors_k, anchors_labels = list(), list(), list()

        for anchor_size in anchor_sizes:
            for i in range(6):
                for j in range(6):
                    y0, x0 = i * stride, j * stride
                    y1, x1 = y0 + anchor_size, x0 + anchor_size
                    if x1 >= 224 or y1 >= 224:
                        continue
                    anchors_q.append(np.array([x0, y0, x1, y1]))
                    x0, y0, x1, y1 = self.anchor_mapping((x0, y0, x1, y1), (l_q, t_q, r_q, b_q), (l_k, t_k, r_k, b_k))
                    if x0 == -1:
                        anchor_label = -1
                        x0 = y0 = 0  # fake anchor
                        x1 = y1 = 1
                    else:
                        anchor_label = 0
                    anchors_labels.append(anchor_label)
                    anchors_k.append(np.array([x0, y0, x1, y1]))
        anchors_q = np.array(anchors_q, dtype=np.float)
        anchors_k = np.array(anchors_k, dtype=np.float)
        anchors_labels = np.array(anchors_labels, dtype=np.float)

        # === C3 ===
        anchor_sizes = [32]
        stride = 24.
        anchors_q_c3, anchors_k_c3, anchors_labels_c3 = list(), list(), list()

        for anchor_size in anchor_sizes:
            for i in range(8):
                for j in range(8):
                    y0, x0 = i * stride, j * stride
                    y1, x1 = y0 + anchor_size, x0 + anchor_size
                    if x1 >= 224 or y1 >= 224:
                        continue
                    anchors_q_c3.append(np.array([x0, y0, x1, y1]))
                    x0, y0, x1, y1 = self.anchor_mapping((x0, y0, x1, y1), (l_q, t_q, r_q, b_q), (l_k, t_k, r_k, b_k))
                    if x0 == -1:
                        anchor_label = -1
                        x0 = y0 = 0  # fake anchor
                        x1 = y1 = 1
                    else:
                        anchor_label = 0
                    anchors_labels_c3.append(anchor_label)
                    anchors_k_c3.append(np.array([x0, y0, x1, y1]))
        anchors_q_c3 = np.array(anchors_q_c3, dtype=np.float)
        anchors_k_c3 = np.array(anchors_k_c3, dtype=np.float)
        anchors_labels_c3 = np.array(anchors_labels_c3, dtype=np.float)

        q, k = self.ending_transform(q), self.ending_transform(k)

        return [q, k], label_weight, [anchors_q, anchors_k, anchors_labels], [anchors_q_c3, anchors_k_c3, anchors_labels_c3]

    @staticmethod
    def anchor_mapping(anchor, loc_q, loc_k, img_size=224.):
        """
        Args:
            anchor: (x0, y0, x1, y1) The coordinates of anchors in image q
            loc_q: (x0, y0, x1, y1) The cropped region of image q w.r.t. the original image
            loc_k: (x0, y0, x1, y1) The cropped region of image k w.r.t. the original image
            img_size (float): Image size
        Returns:
            tuple: (x0, y0, x1, y1) if transformed region falls into the boundary of image k, or (-1, -1, -1, -1) if not
        """
        x0, y0, x1, y1 = anchor
        l_q, t_q, r_q, b_q = loc_q
        l_k, t_k, r_k, b_k = loc_k
        # Step 1: transfer anchor's coordinates (relative to q) to absolute
        l_a = (r_q - l_q) * x0 / img_size + l_q
        r_a = (r_q - l_q) * x1 / img_size + l_q
        t_a = (b_q - t_q) * y0 / img_size + t_q
        b_a = (b_q - t_q) * y1 / img_size + t_q
        # Step 2: transfer anchor's absolute coordinates to relative to k
        x0 = (l_a - l_k) * img_size / (r_k - l_k)
        x1 = (r_a - l_k) * img_size / (r_k - l_k)
        y0 = (t_a - t_k) * img_size / (b_k - t_k)
        y1 = (b_a - t_k) * img_size / (b_k - t_k)
        # Step 3: check whether it's out of boundary
        if x0 < 0 or y0 < 0 or x1 >= img_size or y1 >= img_size:
            x0 = y0 = x1 = y1 = -1
        return x0, y0, x1, y1

    @staticmethod
    def iou(q, k):
        t_q, l_q, b_q, r_q = q
        h_q, w_q = b_q - t_q, r_q - l_q
        t_k, l_k, b_k, r_k = k
        h_k, w_k = b_k - t_k, r_k - l_k
        union = (max(t_q, t_k), max(l_q, l_k), min(b_q, b_k), min(r_q, r_k))
        if union[0] > union[2] or union[1] > union[3]:
            return 0.
        union_size = (union[2] - union[0]) * (union[3] - union[1])
        return union_size / (h_q * w_q + h_k * w_k - union_size)

    @staticmethod
    def ioa(q, k):
        t_q, l_q, b_q, r_q = q
        h_q, w_q = b_q - t_q, r_q - l_q
        t_k, l_k, b_k, r_k = k
        h_k, w_k = b_k - t_k, r_k - l_k
        union = (max(t_q, t_k), max(l_q, l_k), min(b_q, b_k), min(r_q, r_k))
        if union[0] > union[2] or union[1] > union[3]:
            return 0.
        union_size = (union[2] - union[0]) * (union[3] - union[1])
        return union_size / min(h_q * w_q, h_k * w_k)

    def __len__(self):
        return len(self.data)

