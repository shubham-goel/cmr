"""
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags
from tqdm import tqdm
import numpy as np

from ..utils.visualizer import Visualizer
from ..utils.tb_visualizer import TBVisualizer

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir') # Not used!
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')

flags.DEFINE_bool('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as mmomentu')

flags.DEFINE_integer('batch_size', 16, 'Size of minibatches')
flags.DEFINE_integer('num_iter', 0, 'Number of training iterations. 0 -> Use epoch_iter')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 10000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 10, 'save model every k epochs')

## Flags for visualization
flags.DEFINE_integer('display_freq', 1000, 'visuals logging frequency')
flags.DEFINE_boolean('display_visuals', True, 'whether to display images')
flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
flags.DEFINE_boolean('plot_scalars', True, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_integer('display_id', 1, 'Display Id')
flags.DEFINE_integer('display_winsize', 256, 'Display Size')
flags.DEFINE_string('display_server', 'http://0.0.0.0', 'Display server')
flags.DEFINE_integer('display_port', 8097, 'Display port')
flags.DEFINE_integer('display_single_pane_ncols', 0, 'if positive, display all images in a single visdom web panel with certain number of images per row.')


class RecentList():
    def __init__(self,max_size=10):
        assert(max_size > 0)
        self.max_size = max_size
        self.list = []
        self.csum = 0
    def append(self, x):
        if len(self.list) == self.max_size:
            self.csum -= self.list[0]
            self.list.pop(0)
        x = float(x)
        self.list.append(x)
        self.csum += x
    def average(self):
        return self.csum/(len(self.list) + 1e-12)

class AverageMeter():
    def __init__(self, max_size=10):
        self.csum = 0
        self.count = 0
        self.recent = RecentList(max_size=max_size)
    def append(self, x):
        self.csum += float(x)
        self.count += 1
        self.recent.append(x)
    def average(self):
        return self.csum/(self.count + 1e-12)
    def recent_average(self):
        return self.recent.average()

#-------- tranining class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))


    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda(device=gpu_id)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path), strict=False)
        return

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix, gpu_id=self.opts.gpu_id)
        return

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        if opts.use_sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=opts.learning_rate, momentum=opts.beta1)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=opts.learning_rate, betas=(opts.beta1, 0.999))

    def train(self):
        opts = self.opts
        self.smoothed_total_loss = 0
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer

        self.tb_visualizer = TBVisualizer(opts)
        tb_visualizer = self.tb_visualizer

        total_steps = 0
        dataset_size = len(self.dataloader)

        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
            epoch_iter = 0
            loss_meter = AverageMeter()

            pbar = tqdm(self.dataloader,dynamic_ncols=True,total=dataset_size,desc='e{}'.format(epoch))
            for i, batch in enumerate(pbar):
                iter_start_time = time.time()
                self.set_input(batch)
                if not self.invalid_batch:
                    self.optimizer.zero_grad()
                    self.forward()
                    self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.total_loss.item()
                    if opts.is_train:
                        # self.pred_v_copy.retain_grad()
                        # assert(self.pred_v_copy.retains_grad)
                        self.total_loss.backward()
                        # pdb.set_trace()
                        self.optimizer.step()

                total_steps += 1
                epoch_iter += 1

                if opts.display_visuals and (total_steps % opts.display_freq == 0):
                    iter_end_time = time.time()
                    # print('time/itr %.2g' % ((iter_end_time - iter_start_time)/opts.display_freq))
                    visuals = self.get_current_visuals()
                    visualizer.display_current_results(visuals, epoch)
                    visualizer.plot_current_points(self.get_current_points())
                    # tb_visualizer.display_current_results({'img':visuals}, total_steps)

                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    # visualizer.print_current_scalars(epoch, epoch_iter, scalars)
                    if opts.plot_scalars:
                        visualizer.plot_current_scalars(epoch, float(epoch_iter)/dataset_size, opts, scalars)

                loss_meter.append(self.total_loss.item())
                pbar.set_postfix(iter=total_steps, loss_avg=loss_meter.average(), loss_recent=loss_meter.recent_average())


                if opts.is_train and (total_steps % opts.save_latest_freq) == 0:
                    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return

            if opts.is_train and ((epoch+1) % opts.save_epoch_freq == 0):
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)

            if opts.save_camera_pose_dict:

                if opts.render_mean_shape_only:
                    nodeform = '_noDeform'
                else:
                    nodeform = ''

                if (len(self.datasetCameraPoseDict)!=len(self.dataloader.dataset)):
                    print('Saving cam_pose_dict of size ',len(self.datasetCameraPoseDict),'!=',len(self.dataloader.dataset))
                path = 'birds/cmr_'+opts.split+'_'+opts.name+nodeform+'_campose_'+str(epoch)
                stats = {'campose':self.datasetCameraPoseDict}
                np.savez(path, **stats)

                path = 'birds/cmr_'+opts.split+'_'+opts.name+'_maskiou_'+str(epoch)
                stats = {'maskiou':self.datasetMaskIouDict}
                np.savez(path, **stats)

                path = 'birds/cmr_'+opts.split+'_'+opts.name+'_noDeform_maskiou_'+str(epoch)
                stats = {'maskiou':self.datasetMaskIouMsDict}
                np.savez(path, **stats)
                
                path = 'birds/cmr_'+opts.split+'_'+opts.name+'_GtCam_maskiou_'+str(epoch)
                stats = {'maskiou':self.datasetMaskIouGtCamDict}
                np.savez(path, **stats)

                path = 'birds/cmr_'+opts.split+'_'+opts.name+'_noDeform_GTCam_maskiou_'+str(epoch)
                stats = {'maskiou':self.datasetMaskIouMsGtCamDict}
                np.savez(path, **stats)
