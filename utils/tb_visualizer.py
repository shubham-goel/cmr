import numpy as np
import os
import ntpath
import time
import visdom
from . import visutil as util
# from . import html
# from .logger import Logger
import os
from datetime import datetime
import os.path as osp
import pdb
import torch
import tensorboardX

class TBVisualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.log_dir = osp.join(opt.cache_dir, 'logs', opt.name)
        self.stats_dir = self.log_dir + '/stats/'
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not osp.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        log_name = datetime.now().strftime('%H_%M_%d_%m_%Y')
        print("Logging to {}".format(log_name))
        self.display_id = opt.display_id
        self.use_html = opt.is_train and opt.use_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        # self.viz = Logger(self.log_dir,opt.name)
        self.viz = tensorboardX.SummaryWriter(self.log_dir)
        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self.viz.close()

    def plot_images(self, images, global_step):
        for label, image in images.items():
            assert(np.isfinite(image).all()), label
            if(len(image.shape) == 2):
                dataformats = 'HW'
                self.viz.add_image(label,image, global_step, dataformats=dataformats)
            elif(len(image.shape) == 3):
                dataformats = 'HWC' if (image.shape[2]==3) else 'CHW'
                self.viz.add_image(label,image, global_step, dataformats=dataformats)
            elif(len(image.shape) == 4):
                dataformats = 'NHWC' if (image.shape[3]==3) else 'NCHW'
                self.viz.add_images(label,image, global_step, dataformats=dataformats)
            else:
                raise NotImplementedError


    def plot_videos(self, videos, global_step, fps=4):
        for label, video in videos.items():
            assert(np.isfinite(video).all()), label
            if(len(video.shape) == 4): # t,C,H,W
                assert video.shape[1]==3, 'Invalid video shape:{}'.format(video.shape)
                self.viz.add_video(label, video[None], global_step, fps=fps)
            elif(len(image.shape) == 5):
                assert video.shape[2]==3, 'Invalid video shape:{}'.format(video.shape)
                self.viz.add_video(label, video, global_step, fps=fps)
            else:
                raise NotImplementedError

    def plot_meshes(self, meshes, global_step):
        for label, mesh in meshes.items():
            vert = mesh['v']
            assert(torch.isfinite(vert).all()), label
            face = mesh['f'] if 'f' in mesh else None
            color = mesh['c'] if 'c' in mesh else None
            config = mesh['cfg'] if 'cfg' in mesh else {}
            self.viz.add_mesh(label,vert,colors=color,faces=face,config_dict=config,global_step=global_step)

    def plot_embeddings(self, embeddings, global_step):
        for label, embed in embeddings.items():
            if isinstance(embed,dict):
                mat = embed['mat']
                metadata = embed['metadata'] if 'metadata' in embed else None
                metadata_header = embed['metadata_header'] if 'metadata_header' in embed else None
                label_img = embed['label_img'] if 'label_img' in embed else None
                self.viz.add_embedding(mat,tag=label,global_step=global_step,metadata=metadata, label_img=label_img, metadata_header=metadata_header)
            else:
                assert(torch.isfinite(embed).all()), label
                self.viz.add_embedding(embed,tag=label,global_step=global_step)

    def plot_histograms(self, histograms, global_step):
        for label, hist in histograms.items():
            if isinstance(hist,dict):
                values = hist['values']
                bins = hist['bins'] if 'bins' in hist else 'tensorflow'
                max_bins = hist['max_bins'] if 'max_bins' in hist else None
                self.viz.add_histogram(label,values,global_step=global_step,bins=bins, max_bins=max_bins)
            else:
                assert(torch.isfinite(hist).all()), label
                self.viz.add_histogram(label,hist,global_step=global_step)

    def plot_texts(self, texts, global_step):
        for label, text in texts.items():
            self.viz.add_text(label,text,global_step=global_step)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, global_step):
        if 'img' in visuals:
            self.plot_images(visuals['img'], global_step)
        if 'image' in visuals:
            self.plot_images(visuals['image'], global_step)

        if 'video' in visuals:
            fps = visuals['video_fps'] if 'video_fps' in visuals else 4
            self.plot_videos(visuals['video'], global_step, fps=fps)

        if 'mesh' in visuals:
            self.plot_meshes(visuals['mesh'], global_step)

        if 'embed' in visuals:
            self.plot_embeddings(visuals['embed'], global_step)

        if 'hist' in visuals:
            self.plot_histograms(visuals['hist'], global_step)

        if 'text' in visuals:
            self.plot_texts(visuals['text'], global_step)

        if 'scalar' in visuals:
            self.plot_current_scalars(global_step, None, visuals['scalar'])

    def save_raw_stats(self, stats, name, epoch):
        path = '{}/{}_{}'.format(self.stats_dir, name, epoch)
        np.savez(path, **stats)

    def log_trj(self, global_step, trj_pred, trj_gt, trj_mask, classify_trj=False):
        # tps = []
        # tgts = []
        # for tp, tgt,tm  in zip(trj_pred, trj_gt, trj_mask):
        #     tps.append(torch.exp(tp).view(-1))
        #     tgts.append((tgt).view(-1))
        #
        # tps = torch.cat(tps).data
        # tgts = torch.cat(tgts).data
        # self.viz.histo_summary('trajectory/preds', tps.cpu().numpy().reshape((-1)), global_step)
        # self.viz.histo_summary('trajectory/gts', tgts.cpu().numpy().reshape((-1)), global_step)
        if classify_trj:
            self.hist_summary_trj(global_step, 'trajectory/preds_probs', trj_pred, trj_mask, use_exp=classify_trj)
        else:
            self.hist_summary_trj(global_step, 'trajectory/preds', trj_pred, trj_mask, use_exp=classify_trj)
        self.hist_summary_trj(global_step, 'trajectory/gts', trj_gt, trj_mask)
        return

    def hist_summary_list(self, global_step, tag, data_list):
        t = []
        for l in data_list:
            t.append(l.view(-1))
        t = torch.cat(t)
        self.viz.add_histogram(tag, t.cpu().numpy().reshape(-1), global_step)

    def log_pwd(self, global_step, pwd_pred, pwd_gt):
        self.hist_summary_list(global_step, 'pwd/pred', pwd_pred)
        self.hist_summary_list(global_step, 'pwd/gt', pwd_gt)
        return

    def log_histogram(self, global_step, log_dict):
        for tag, value in log_dict.items():
            self.viz.add_histogram(tag, value.data.cpu().numpy(), global_step)
        return

    def log_trj_max_indices(self, global_step, trj_pred, trj_mask, classify_trj=False):
        ts = []
        for t, tm in zip(trj_pred, trj_mask):
            n_elements = torch.sum(tm).data[0]
            if n_elements > 0:
                indices = torch.nonzero(torch.ge(tm.view(-1), 0.5).data).view(-1)
                # pdb.set_trace()
                t_selected = t.view(-1, t.size(-1))
                t_selected = t_selected[indices]
                t_selected = t_selected.max(dim=-1)[1]
                ts.append(t_selected.view(-1))

        if len(ts) > 0:
            ts = torch.cat(ts)
            ts = ts.data
            self.viz.add_histogram('trajectory/preds', ts.cpu().numpy().reshape((-1)), global_step)
        return


    def hist_summary_trj(self, global_step, name, trj, trj_mask, use_exp=False):
        ts = []

        for t, tm  in zip(trj, trj_mask):
            n_elements = torch.sum(tm).data[0]
            if n_elements > 0:
                indices = torch.nonzero(torch.ge(tm.view(-1), 0.5).data).view(-1)
                t_selected = t.view(-1)[indices]
                if use_exp:
                    ts.append(torch.exp(t_selected).view(-1))
                else:
                    ts.append(t_selected.view(-1))

        if len(ts) > 0:
            ts = torch.cat(ts)
            ts = ts.data
            self.viz.add_histogram(name, ts.cpu().numpy().reshape((-1)), global_step)
        return

    def log_adj_matrix(self, global_step, adj_matrix, adj_mask, non_masked):
        self.viz.add_histogram('adjacency/values', adj_matrix.cpu().numpy().reshape((-1)), global_step)
        self.viz.add_histogram('adjacency/mask', adj_mask.cpu().numpy().reshape((-1)), global_step)
        self.viz.add_histogram('adjacency/non_masked_values', non_masked.cpu().numpy().reshape((-1)), global_step)

    # scalars: dictionary of scalar labels and values
    # def plot_current_scalars(self, epoch, counter_ratio, opt, scalars):
    #
    #
    #
    #
    #     if not hasattr(self, 'plot_data'):
    #         self.plot_data = {'X':[],'Y':[], 'legend':list(scalars.keys())}
    #     self.plot_data['X'].append(epoch + counter_ratio)
    #     self.plot_data['Y'].append([scalars[k] for k in self.plot_data['legend']])
    #     self.vis.line(
    #         X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
    #         Y=np.array(self.plot_data['Y']),
    #         opts={
    #             'title': self.name + ' loss over time',
    #             'legend': self.plot_data['legend'],
    #             'xlabel': 'epoch',
    #             'ylabel': 'loss'},
    #         win=self.display_id)

    def plot_current_scalars(self,global_step, opt, scalars):
        for key, value in scalars.items():
            if isinstance(value, dict):
                self.viz.add_scalars(key, value, global_step)
            else:
                self.viz.add_scalar(key, value, global_step)

    # scatter plots
    def plot_current_points(self, points, disp_offset=10):
        idx = disp_offset
        for label, pts in points.items():
            #image_numpy = np.flipud(image_numpy)
            self.vis.scatter(
                pts, opts=dict(title=label, markersize=1), win=self.display_id + idx)
            idx += 1

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars, start_time=None):
        if start_time is None:
            message = '(time : %.3f, epoch: %d, iters: %d) ' % (epoch, i)
        else:
            time_diff = (time.time() - start_time)
            message = '(time : %.2f, epoch: %d, iters: %d) ' % (time_diff, epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals['img'].items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
