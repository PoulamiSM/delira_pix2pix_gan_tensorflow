import tensorflow as tf
import os
import numpy as np
import argparse
from delira.training import Parameters

from delira.models.gan.lpips_score import lpips_score

parser = argparse.ArgumentParser(description='')
parser.add_argument('--image_size', dest='image_size', default=256) # output image size
parser.add_argument('--resize_size', dest='resize_size', default=1024)  # dimension to which input image is resized before cropping
parser.add_argument('--input_c_dim', dest='input_c_dim', default=1)  # input channel dimension
parser.add_argument('--output_c_dim', dest='output_c_dim', default=4)  # output channel dimension
parser.add_argument('--df_dim', dest='df_dim', default=64) # number of filters in the first layer of discriminator
parser.add_argument('--gf_dim', dest='gf_dim', default=64) # number of filters in the first layer of generator
parser.add_argument('--f_size', dest='f_size', default=5) # size of kernel
parser.add_argument('--strides', dest='strides', default=2) # size of strides
parser.add_argument('--l1_lambda', dest='l1_lambda', default=80) # scaling the L1 loss factor
parser.add_argument('--batch_size', dest='batch_size', default=4)  # batchsize to use
parser.add_argument('--val_batch_size', dest='val_batch_size', default=1) # batchsize to use for validation
parser.add_argument('--num_epochs', dest='num_epochs', default=300) # number of epochs to train
parser.add_argument('--learning_rate', dest='learning_rate', default=2e-6)
parser.add_argument('--dataset_name', dest='dataset_name', default='Fabric25') # Training Dataset
parser.add_argument('--val_dataset', dest='val_dataset', default='Fabric21') # Validation Dataset
parser.add_argument('--exp_name', dest='exp_name', default='issue_2')
parser.add_argument('--file_name', dest='file_name', default='')
parser.add_argument('--rot_angle', dest='rot_angle', default=0.25*np.pi) # For rotation experiments using spatial transform
args = parser.parse_args()


params = Parameters(fixed_params={
    "model": {
        "image_size": int(args.image_size),
        "input_c_dim": args.input_c_dim,
        "output_c_dim": args.output_c_dim,
        "df_dim": args.df_dim,
        "gf_dim": args.gf_dim,
        "f_size": args.f_size,
        "strides": args.strides,
        "l1_lambda": args.l1_lambda
    },
    "training": {
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "num_epochs": args.num_epochs,

        "optimizer_cls": tf.train.AdamOptimizer, # optimization algorithm to use
        "optimizer_params": {'learning_rate': args.learning_rate,
                             #'momentum': 0.5
                             'beta1': 0.5,
                             #'beta2': 0.8
                            },
        "criterions": {'CE': tf.losses.sigmoid_cross_entropy},
        "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
        "lr_sched_params": {}, # the corresponding initialization parameters
        "metrics": {'LPIPS': lpips_score([None, 3, args.image_size, args.image_size])}, # and some evaluation metrics
        "dataset_name": args.dataset_name,
        "val_dataset": args.val_dataset,
        "exp_name": args.exp_name
    }
})



### Path to log the checkpoints is set here

from trixi.logger.tensorboard.tensorboardxlogger import TensorboardXLogger
from delira.logging import TrixiHandler
import logging

exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(params.nested_get("dataset_name"),
                                         params.nested_get("val_dataset"), params.nested_get("image_size"),
                                         params.nested_get("batch_size"), params.nested_get("num_epochs"),
                                         params.nested_get("learning_rate"), args.file_name)

log_path = '/work/scratch/poulami/TB_logs/Delira/cGAN'
logger_cls = TensorboardXLogger
logging.basicConfig(level=logging.INFO,
                    handlers=[TrixiHandler(logger_cls, 0, os.path.join(log_path, params.nested_get("exp_name"), exp_name + '/'))])

logger = logging.getLogger("Test Logger")


### Data loading


from delira.data_loading import ConditionalGanDataset
from delira.data_loading.load_fn import load_sample_cgan

root_path = '/work/scratch/poulami/Dataset/LW Data/'
#path_train = os.path.join(root_path, params.nested_get("dataset_name") + '/')
path_train = os.path.join(root_path, params.nested_get("dataset_name") + '/Original50mm/') # path for LW data
'''
# For multiple fabric types
train_dataset = ['Fabric7', 'Fabric8', 'Fabric26', 'Fabric25']
path_train = []
for i in train_dataset:
    dataset_name = i
    path_train.append(os.path.join(root_path, dataset_name + '/Original50mm/'))'''
#path_val = os.path.join(root_path, params.nested_get("val_dataset") + '/')
path_val = os.path.join(root_path, params.nested_get("val_dataset") + '/Original50mm/') # path for LW data
'''
# For multiple fabric types
val_dataset = ['Fabric7', 'Fabric8', 'Fabric21', 'Fabric26', 'Fabric25']
path_val = []
for i in val_dataset:
    dataset_name = i
    path_val.append(os.path.join(root_path, dataset_name + '/Original50mm/'))'''


dataset_train = ConditionalGanDataset(path_train, load_sample_cgan, ['.PNG', '.png'], ['.PNG', '.png'],
                                      #train_split=True, split=0.7, data_train=True
                                      )


dataset_val = ConditionalGanDataset(path_val, load_sample_cgan, ['.PNG', '.png'], ['.PNG', '.png'],
                                    #train_split=True, split=0.7, data_train=False
                                    )



### Transforms applied to data

from batchgenerators.transforms import RandomCropTransform, Compose
from batchgenerators.transforms.spatial_transforms import ResizeTransform,SpatialTransform


transforms = Compose([
    #SpatialTransform(patch_size=(1024, 1024), do_rotation=True, patch_center_dist_from_border=1024, border_mode_data='reflect',
     #                border_mode_seg='reflect', angle_x=(args.rot_angle, args.rot_angle), angle_y=(0, 0), angle_z=(0, 0),
      #               do_elastic_deform=False, order_data=1, order_seg=1)
    ResizeTransform((int(args.resize_size), int(args.resize_size)), order=1),
    RandomCropTransform((params.nested_get("image_size"), params.nested_get("image_size"))),
    ])


###  Data manager

from delira.data_loading import BaseDataManager, SequentialSampler, RandomSampler

manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"),
                                transforms=transforms,
                                sampler_cls=RandomSampler,
                                n_process_augmentation=4)

manager_val = BaseDataManager(dataset_val, params.nested_get("val_batch_size"),
                              transforms=transforms,
                              sampler_cls=SequentialSampler,
                              n_process_augmentation=4)

import warnings
warnings.simplefilter("ignore", UserWarning) # ignore UserWarnings raised by dependency code
warnings.simplefilter("ignore", FutureWarning) # ignore FutureWarnings raised by dependency code




### Experiment setup for training

from delira.training import TfExperiment
from delira.training.train_utils import create_optims_gan_default_tf
from delira.models.gan import ConditionalGenerativeAdversarialNetworkBaseTf




experiment = TfExperiment(params, ConditionalGenerativeAdversarialNetworkBaseTf,
                          name=exp_name,
                          save_path=os.path.join("/work/scratch/poulami/Delira_experiments/cGAN",
                                                  params.nested_get("exp_name")),
                          optim_builder=create_optims_gan_default_tf,
                          gpu_ids=[0], val_score_key='val_LPIPS_BL_mean', val_score_mode='lowest')


model = experiment.run(manager_train, manager_val)




