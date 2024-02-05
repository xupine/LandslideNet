class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'PATH/VOC2012/'
        elif dataset == 'sbd':
            return 'PATH/benchmark_RELEASE/'
        elif dataset == 'vaihingen':
            return 'PATH/Vaihingen/'
        elif dataset == 'cityscapes':
            return 'PATH/cityscapes/' 
        elif dataset == 'coco':
            return 'PATH/coco/'
        elif dataset == 'potsdam':
            return 'PATH//Potsdam/'
        elif dataset == 'landslide':
            return 'PATH/landslide/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
