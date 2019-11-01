"""
Demo of CMR.

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python -m cmr.demo --name bird_net --num_train_epoch 500 --img_dir_path cmr/demo_data/0000000000.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, os.path

from absl import flags, app
import numpy as np
import skimage.io as io

import torch
import cv2

from .nnutils import test_utils
from .nnutils import predictor as pred_util
from .utils import image as img_util


flags.DEFINE_string('bird_id', 'KpJUvl2ekzo_0009__bird00000248', 'Images to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS


def read_file_as_list(fname):
    with open(fname, 'r') as f:
        ll = f.readlines()
    return [l.strip() for l in ll]

def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts'][0]
    cam = outputs['cam_pred'][0]
    texture = outputs['texture'][0]
    shape_pred = renderer(vert, cam)
    img_pred = renderer(vert, cam, texture=texture)

    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp3 = renderer.diff_vp(
        vert, cam, angle=60, axis=[1, 0, 0], texture=texture)

    img = np.transpose(img, (1, 2, 0))
    
    # import ipdb; ipdb.set_trace()
    row0 = np.concatenate((img[:256,:256,:], shape_pred/255., img_pred/255.), axis=1)
    row1 = np.concatenate((vp1/255., vp2/255., vp3/255.), axis=1)
    img_all = np.concatenate((row0, row1), axis=0)

    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.imshow(img_all)
    # plt.axis('off')
    # plt.show()
    # plt.waitforbuttonpress()

    img_all = img_all[:,:,::-1]
    return img_all

    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(231)
    # plt.imshow(img)
    # plt.title('input')
    # plt.axis('off')
    # plt.subplot(232)
    # plt.imshow(shape_pred)
    # plt.title('pred mesh')
    # plt.axis('off')
    # plt.subplot(233)
    # plt.imshow(img_pred)
    # plt.title('pred mesh w/texture')
    # plt.axis('off')
    # plt.subplot(234)
    # plt.imshow(vp1)
    # plt.title('different viewpoints')
    # plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(vp2)
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(vp3)
    # plt.axis('off')
    # plt.draw()
    # plt.show()
    # plt.waitforbuttonpress()

def main(_):
    predictor = pred_util.MeshPredictor(opts)

    data_proc_dir = '/data3/shubham/birds_data/proc/masked/'
    data_viz_dir = '/data3/shubham/birds_data/info/cmr/'

    split_id = opts.bird_id
    frames_dir = data_proc_dir + split_id + '/frame/'
    output_save_dir = data_viz_dir + split_id + '/'
    if not os.path.isdir(output_save_dir):
        os.mkdir(output_save_dir)
    num_images = len([name for name in os.listdir(frames_dir) if os.path.isfile('%s/%s'%(frames_dir, name))])

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_writer = cv2.VideoWriter(data_viz_dir + split_id + '.avi', fourcc, 25, (256*3, 256*3))

    for iii in range(num_images):
        img_path = '%s/%010d.jpg'%(frames_dir, iii)
        save_path = '%s/%010d.jpg'%(output_save_dir, iii)

        img = preprocess_image(img_path, img_size=opts.img_size)

        batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

        outputs = predictor.predict(batch)

        # This is resolution
        renderer = predictor.vis_rend
        renderer.set_light_dir([0, 1, -1], 0.4)

        img_out = visualize(img, outputs, predictor.vis_rend)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',img_out)
        cv2.waitKey(1)
        cv2.imwrite(save_path, img_out*255)

    #     vid_writer.write(img_out)
    # vid_writer.release()

if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
