from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms1 as tr

class Landslide(Dataset):

    NUM_CLASSES = 2
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('landslide_t'),
                 split='train',
                 max_iters=None,
                 ):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir,split, 'src')
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        self.im_ids = []
        self.images = []
        n=len([name for name in os.listdir(self._image_dir) if os.path.isfile(os.path.join(self._image_dir, name))])
        print(self.split,n)
        for i in range(n):
            i=str(i)
            _image = os.path.join(self._image_dir, i + ".jpg")
            assert os.path.isfile(_image)
            self.im_ids.append(i)
            self.images.append(_image)

        if not max_iters==None:
            self.im_ids = self.im_ids * int(np.ceil(float(max_iters) / len(self.im_ids)))

            self.images = self.images * int(np.ceil(float(max_iters) / len(self.images)))

        print(self.split, len(self.images))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img = self._make_img_gt_point_pair(index)
        sample = {'image': _img}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')

        return _img

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Landslide(split=' + str(self.split) + ')'      

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512

    Landslide_train = Landslide(args, split='train')

    dataloader = DataLoader(Landslide_train, batch_size=10, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='landslide')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)