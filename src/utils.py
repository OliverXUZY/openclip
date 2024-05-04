import os
def set_gpu(gpu):
    print('set gpu: {:s}'.format(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu