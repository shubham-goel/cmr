"""
Script for the bird shape, pose and texture experiment.

The model takes imgs, outputs the deformation to the mesh & camera parameters
Loss consists of:
- keypoint reprojection loss
- mask reprojection loss
- smoothness/laplacian priors on triangles
- texture reprojection losses

example usage : python -m cmr.experiments.shape --name=bird_shape --plot_scalars --save_epoch_freq=1 --batch_size=8 --display_visuals --display_freq=2000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import os.path as osp
import numpy as np
import torch
torch.backends.cudnn.benchmark = True

import torchvision
from torch.autograd import Variable
import scipy.io as sio
from collections import OrderedDict

from ..data import cub as cub_data
from ..utils import visutil
from ..utils import bird_vis
from ..utils import image as image_utils
from ..nnutils import train_utils
from ..nnutils import loss_utils
from ..nnutils import mesh_net
from ..nnutils.nmr import NeuralRenderer_pytorch as NeuralRenderer
from ..nnutils.nmr import SoftRas
from ..nnutils import geom_utils

flags.DEFINE_string('dataset', 'cub', 'cub or pascal or p3d')
# Weights:
flags.DEFINE_float('kp_loss_wt', 30., 'keypoint loss weight')
flags.DEFINE_float('mask_loss_wt', 2., 'mask loss weight')
flags.DEFINE_float('cam_loss_wt', 2., 'weights to camera loss')
flags.DEFINE_float('deform_reg_wt', 10., 'reg to deformation')
flags.DEFINE_float('template_deform_reg_wt', 5., 'reg to deformation')
flags.DEFINE_float('triangle_reg_wt', 30., 'weights to triangle smoothness prior')
flags.DEFINE_float('vert2kp_loss_wt', .16, 'reg to vertex assignment')
flags.DEFINE_float('tex_loss_wt', .5, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_boolean('use_gtpose', True, 'if true uses gt pose for projection, but camera still gets trained.')
flags.DEFINE_enum('renderer', 'nmr', ['nmr','softras'], 'What renderer to use')
flags.DEFINE_boolean('texture_detach_shape', True, 'if true detach shape, camera while predicting texture.')
flags.DEFINE_boolean('save_camera_pose_dict', False, 'Save camera pose dict for entire dataset to file')
flags.DEFINE_boolean('save_maskiou_dict', False, 'Save maskiou dict for entire dataset to file')
flags.DEFINE_boolean('render_mean_shape_only', False, 'Render mean shape only')
flags.DEFINE_float('delta_v_scale', 1, 'scale for delta_v')
flags.DEFINE_boolean('nokp', False, 'No-keypoints')

flags.DEFINE_string('shape_path', 'birds/csm_mesh/bird_mean_shape.npy', 'Path to initial mean shape')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # ----------
        # Options
        # ----------
        self.symmetric = opts.symmetric
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        # sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri']-1)

        mean_shape = np.load(opts.shape_path,allow_pickle=True,encoding='latin1').item()
        verts_uv = mean_shape['verts_uv']
        verts = mean_shape['verts']
        faces = mean_shape['faces']
        self.verts_uv = torch.from_numpy(verts_uv).float() # V,3
        self.verts = torch.from_numpy(verts).float() # V,3
        self.faces = torch.from_numpy(faces).long()  # F,2

        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, verts, faces, verts_uv, nz_feat=opts.nz_feat, num_kps=opts.num_kps)

        self.template_shape = self.model.get_mean_shape().detach().cuda()

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        if not opts.is_train:
            self.model.eval()

        self.model = self.model.cuda(device=opts.gpu_id)

        # Data structures to use for triangle priors.
        edges2verts = self.model.edges2verts
        # B x E x 4
        edges2verts = np.tile(np.expand_dims(edges2verts, 0), (opts.batch_size, 1, 1))
        self.edges2verts = Variable(torch.LongTensor(edges2verts).cuda(device=opts.gpu_id), requires_grad=False)
        # For renderering.
        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        if opts.renderer == 'nmr':
            renderer_class = NeuralRenderer
        else:
            renderer_class = SoftRas

        self.renderer = renderer_class(opts.img_size)
        self.renderer_predcam = renderer_class(opts.img_size) #for camera loss via projection

        # Need separate NMR for each fwd/bwd call.
        if opts.texture:
            self.tex_renderer = renderer_class(opts.img_size, light_intensity_ambient=1.0, light_intensity_directionals=0.0)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        # For visualization
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces.data.cpu().numpy())

        # Save cameras
        self.datasetCameraPoseDict = {}
        self.datasetMaskIouDict = {}
        self.datasetMaskIouMsDict = {}
        self.datasetMaskIouGtCamDict = {}
        self.datasetMaskIouMsGtCamDict = {}
        return

    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            self.data_module = cub_data
        else:
            print('Unknown dataset %d!' % opts.dataset)

        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        self.projection_loss = loss_utils.kp_l2_loss
        self.mask_loss_fn = torch.nn.MSELoss()
        self.entropy_loss = loss_utils.entropy_loss
        self.deform_reg_fn = loss_utils.deform_l2reg
        self.camera_loss = loss_utils.camera_loss
        self.triangle_loss_fn = loss_utils.LaplacianLoss(self.faces)

        if self.opts.texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss()
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss


    def set_input(self, batch):
        opts = self.opts

        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        kp_tensor = batch['kp'].type(torch.FloatTensor)
        cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
        self.frame_id = batch['inds'].type(torch.LongTensor).cuda()

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.masks = Variable(
            mask_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.kps = Variable(
            kp_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.cams = Variable(
            cam_tensor.cuda(device=opts.gpu_id), requires_grad=False)

        # Compute barrier distance transform.
        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in batch['mask']])
        dt_tensor = torch.FloatTensor(mask_dts).cuda(device=opts.gpu_id)
        # B x 1 x N x N
        self.dts_barrier = Variable(dt_tensor, requires_grad=False).unsqueeze(1)

        # Complete batch
        if self.input_imgs.shape[0]!=opts.batch_size:
            indices = torch.arange(opts.batch_size, dtype=torch.long, device=self.input_imgs.device)
            indices[indices>=self.input_imgs.shape[0]] = self.input_imgs.shape[0]-1
            self.frame_id = torch.index_select(self.frame_id, 0, indices)
            self.input_imgs = torch.index_select(self.input_imgs, 0, indices)
            self.imgs = torch.index_select(self.imgs, 0, indices)
            self.masks = torch.index_select(self.masks, 0, indices)
            self.kps = torch.index_select(self.kps, 0, indices)
            self.cams = torch.index_select(self.cams, 0, indices)
            self.dts_barrier = torch.index_select(self.dts_barrier, 0, indices)

    def forward(self):
        opts = self.opts
        if opts.texture:
            pred_codes, self.textures = self.model.forward(self.input_imgs)
        else:
            pred_codes = self.model.forward(self.input_imgs)
        self.delta_v, scale, trans, quat = pred_codes

        self.cam_pred = torch.cat([scale, trans, quat], 1)

        if opts.only_mean_sym:
            del_v = self.delta_v
        else:
            del_v = self.model.symmetrize(self.delta_v)

        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()
        if opts.render_mean_shape_only:
            self.pred_v = self.mean_shape + del_v*0
        else:
            self.pred_v = self.mean_shape + del_v*opts.delta_v_scale

        # Compute keypoints.
        self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Decide which camera to use for projection.
        if opts.use_gtpose and (opts.nokp==False):
            proj_cam = self.cams
        else:
            proj_cam = self.cam_pred

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts, proj_cam)

        # Render mask.
        self.mask_pred = self.renderer.forward(self.pred_v, self.faces.int(), proj_cam)
        if opts.save_maskiou_dict:
            self.mask_pred_ms = self.renderer.forward(self.mean_shape + del_v*0, self.faces.int(), proj_cam)
            self.mask_pred_gtcam = self.renderer.forward(self.pred_v, self.faces.int(), self.cams)
            self.mask_pred_ms_gtcam = self.renderer.forward(self.mean_shape + del_v*0, self.faces.int(), self.cams)

        if opts.texture:
            self.texture_flow = self.textures
            self.textures = geom_utils.sample_textures(self.texture_flow, self.imgs)
            tex_size = self.textures.size(2)
            self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)

            if opts.texture_detach_shape:
                self.texture_pred = self.tex_renderer.forward(self.pred_v.detach(), self.faces.int(), proj_cam.detach(), textures=self.textures)
            else:
                self.texture_pred = self.tex_renderer.forward(self.pred_v, self.faces.int(), proj_cam, textures=self.textures)
        else:
            self.textures = None

        # Compute losses for this instance.
        if not opts.nokp:
            self.kp_loss = self.projection_loss(self.kp_pred, self.kps)
            self.cam_loss = self.camera_loss(self.cam_pred, self.cams, 0)
        else:
            self.kp_loss = torch.tensor(0, dtype=torch.float, device=self.input_imgs.device)
            self.cam_loss = torch.tensor(0, dtype=torch.float, device=self.input_imgs.device)
        self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks)

        self.mask_iou = loss_utils.maskiou(self.mask_pred, self.masks)
        if opts.save_maskiou_dict:
            self.mask_iou_ms = loss_utils.maskiou(self.mask_pred_ms, self.masks)
            self.mask_iou_gtcam = loss_utils.maskiou(self.mask_pred_gtcam, self.masks)
            self.mask_iou_ms_gtcam = loss_utils.maskiou(self.mask_pred_ms_gtcam, self.masks)
        else:
            self.mask_iou_ms = self.mask_iou
            self.mask_iou_gtcam = self.mask_iou
            self.mask_iou_ms_gtcam = self.mask_iou

        if opts.texture:
            self.tex_loss = self.texture_loss(self.texture_pred, self.imgs, self.mask_pred, self.masks)
            self.tex_dt_loss = self.texture_dt_loss_fn(self.texture_flow, self.dts_barrier)
        else:
            self.tex_dt_loss = torch.zeros_like(self.cam_loss)

        # Priors:
        if not opts.nokp:
            self.vert2kp_loss = self.entropy_loss(self.vert2kp)
        else:
            self.vert2kp_loss = torch.tensor(0, dtype=torch.float, device=self.input_imgs.device)
        self.deform_reg = self.deform_reg_fn(self.delta_v)

        template_deformation = self.pred_v - self.template_shape[None]
        self.template_deform_reg = self.deform_reg_fn(template_deformation)

        self.pred_v_copy = self.pred_v + 0
        self.triangle_loss = self.triangle_loss_fn(self.pred_v_copy)

        # Finally sum up the loss.
        # Instance loss:
        self.total_loss = torch.tensor(0, dtype=torch.float, device=self.input_imgs.device)
        self.total_loss += opts.mask_loss_wt * self.mask_loss
        if opts.texture:
            self.total_loss += opts.tex_loss_wt * self.tex_loss
        if not opts.nokp:
            self.total_loss += opts.kp_loss_wt * self.kp_loss
            self.total_loss += opts.cam_loss_wt * self.cam_loss

        # Priors:
        if not opts.nokp:
            self.total_loss += opts.vert2kp_loss_wt * self.vert2kp_loss
        self.total_loss += opts.deform_reg_wt * self.deform_reg
        self.total_loss += opts.template_deform_reg_wt * self.template_deform_reg
        self.total_loss += opts.triangle_reg_wt * self.triangle_loss

        self.total_loss += opts.tex_dt_loss_wt * self.tex_dt_loss


        if opts.save_camera_pose_dict:
            cam_pred_multipose = self.cam_pred[:,None,:]
            quat_score = torch.ones_like(cam_pred_multipose[:,:,0])
            self.update_camera_pose_dict(self.frame_id.squeeze(-1), cam_pred_multipose, quat_score, self.cams[:,:3],
                self.mask_iou[:,None], self.mask_iou_ms[:,None], self.mask_iou_gtcam[:,None], self.mask_iou_ms_gtcam[:,None],)


    def update_camera_pose_dict(self, frameids, cams, scores, gt_st, maskious, maskious_ms, maskious_gtcam, maskious_ms_gtcam):
        ## Dictionary of all camera poses
        # dict[frameid] = (Px7: [scale trans quat], P:score, 3:gtscale gttrans)
        assert(frameids.shape[0] == cams.shape[0] == scores.shape[0] == gt_st.shape[0])
        assert(frameids.dim()==1)
        assert(scores.dim()==2)
        assert(maskious.dim()==2)
        assert(gt_st.dim()==2)
        assert(cams.dim()==3)

        frameids = frameids.detach()
        cams = cams.detach()
        scores = scores.detach()
        gt_st = gt_st.detach()
        maskious = maskious.detach()

        frame_id_isflip = frameids > (int(1e6)-frameids)
        flip_cams = geom_utils.reflect_cam_pose(cams)
        flip_gt_st = gt_st * torch.tensor([1,-1,1], dtype=gt_st.dtype, device=gt_st.device)
        gt_st = torch.where(frame_id_isflip[:,None], flip_gt_st, gt_st)
        cams = torch.where(frame_id_isflip[:,None,None], flip_cams, cams)
        frameids = torch.where(frame_id_isflip, int(1e6)-frameids, frameids)

        for i in range(frameids.shape[0]):
            f = frameids[i].item()
            if f not in self.datasetCameraPoseDict:
                # print(cams[i,:,:].shape)
                # print(scores[i,:].shape)
                # print(gt_st[i,:].shape)
                # print(maskious[i,:].shape)
                # print('')
                self.datasetCameraPoseDict[f] = (cams[i,:,:].cpu(), scores[i,:].cpu(), gt_st[i,:].cpu())
                self.datasetMaskIouDict[f] = maskious[i,:].cpu()
                self.datasetMaskIouMsDict[f] = maskious_ms[i,:].cpu()
                self.datasetMaskIouGtCamDict[f] = maskious_gtcam[i,:].cpu()
                self.datasetMaskIouMsGtCamDict[f] = maskious_ms_gtcam[i,:].cpu()


    def get_current_visuals(self):
        vis_dict = {}
        mask_concat = torch.cat([self.masks, self.mask_pred], 2)


        if self.opts.texture:
            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred.detach()
            # B x H x W x 2
            uv_flows = uv_flows.permute(0, 2, 3, 1).detach()
            uv_images = torch.nn.functional.grid_sample(self.imgs, uv_flows).detach()
            if self.opts.renderer == 'softras':
                bb,f,t = uv_images.size(0),self.faces.shape[1], self.opts.tex_size
                uv_sampler = self.model.uv_sampler_nmr.view(f,t*t,2)[None].expand(bb,-1,-1,-1)
                textures_nmr = torch.nn.functional.grid_sample(uv_images, uv_sampler)
                textures_nmr = textures_nmr.view(bb, -1, f, t, t).permute(0, 2, 3, 4, 1)
                textures_nmr = textures_nmr.unsqueeze(4).repeat(1, 1, 1, 1, t, 1)
            else:
                textures_nmr = self.textures

        num_show = min(2, self.opts.batch_size)
        show_uv_imgs = []
        show_uv_flows = []

        for i in range(num_show):
            input_img = bird_vis.kp2im(self.kps[i].data, self.imgs[i].data)
            pred_kp_img = bird_vis.kp2im(self.kp_pred[i].data, self.imgs[i].data)
            masks = bird_vis.tensor2mask(mask_concat[i].data)
            if self.opts.texture:
                texture_here = textures_nmr[i]
            else:
                texture_here = None

            rend_predcam = self.vis_rend(self.pred_v[i], self.cam_pred[i], texture=texture_here)
            # Render from front & back:
            rend_frontal = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], texture=texture_here, kp_verts=self.kp_verts[i])
            rend_top = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], axis=[0, 1, 0], texture=texture_here, kp_verts=self.kp_verts[i])
            diff_rends = np.hstack((rend_frontal, rend_top))

            # Render GT view from front & back:
            rend_frontal = self.vis_rend.diff_vp(self.pred_v[i], self.cams[i], texture=texture_here, kp_verts=self.kp_verts[i])
            rend_top = self.vis_rend.diff_vp(self.pred_v[i], self.cams[i], axis=[0, 1, 0], texture=texture_here, kp_verts=self.kp_verts[i])
            diff_rends_gt = np.hstack((rend_frontal, rend_top))

            if self.opts.texture:
                uv_img = bird_vis.tensor2im(uv_images[i].data)
                show_uv_imgs.append(uv_img)
                uv_flow = bird_vis.visflow(uv_flows[i].data)
                show_uv_flows.append(uv_flow)

                tex_img = bird_vis.tensor2im(self.texture_pred[i].data)
                imgs = np.hstack((input_img, pred_kp_img, tex_img))
            else:
                imgs = np.hstack((input_img, pred_kp_img))

            rend_gtcam = self.vis_rend(self.pred_v[i], self.cams[i], texture=texture_here)
            rends_gt = np.hstack((rend_gtcam, diff_rends_gt, np.zeros(rend_predcam.shape, dtype=diff_rends_gt.dtype)))
            rends = np.hstack((rend_predcam, diff_rends, np.zeros(rend_predcam.shape, dtype=diff_rends_gt.dtype)))
            if self.opts.texture:
                rends_gt = np.hstack((rends_gt, np.zeros(rend_predcam.shape, dtype=diff_rends_gt.dtype)))
                rends = np.hstack((rends, np.zeros(rend_predcam.shape, dtype=diff_rends_gt.dtype)))

            hh = np.hstack((imgs, masks))
            vis_dict['%d' % i] = np.vstack((hh, rends_gt, rends))
            vis_dict['masked_img_%d' % i] = bird_vis.tensor2im((self.imgs[i] * self.masks[i]).data)

        if self.opts.texture:
            vis_dict['uv_images'] = np.hstack(show_uv_imgs)
            vis_dict['uv_flow_vis'] = np.hstack(show_uv_flows)

        return vis_dict


    def get_current_points(self):
        return {
            'mean_shape': visutil.tensor2verts(self.mean_shape.data),
            'verts': visutil.tensor2verts(self.pred_v.data),
        }

    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.item()),
            ('mask_loss', self.mask_loss.item()),
            ('deform_reg', self.deform_reg.item()),
            ('template_deform_reg', self.template_deform_reg.item()),
            ('tri_loss', self.triangle_loss.item()),
        ])
        sc_dict['kp_loss'] = self.kp_loss.item()
        sc_dict['vert2kp_loss'] = self.vert2kp_loss.item()
        sc_dict['cam_loss'] = self.cam_loss.item()
        if self.opts.texture:
            sc_dict['tex_loss'] = self.tex_loss.item()
            sc_dict['tex_dt_loss'] = self.tex_dt_loss.item()

        return sc_dict


def main(_):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
