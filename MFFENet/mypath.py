class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/media/user/新加卷1/xupine/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'vaihingen':
            return '/home/dell/model_dataset/Vaihingen1/'
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'potsdam':
            return '/home/dell/model_dataset/Potsdam1/'
        elif dataset == 'landslide':
            return '/home/dell/landslide/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
