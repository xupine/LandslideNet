class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'landslide_s':
            return '/home/dell/landslide/source/'
        elif dataset == 'landslide_t':
            return '/home/dell/landslide/target/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
