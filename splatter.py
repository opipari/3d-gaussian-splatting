import os
import math
import torch
import torch.nn as nn
import gaussian
from utils import read_points3d_binary, read_cameras_binary, read_images_binary, q2r, jacobian_torch, initialize_sh, inverse_sigmoid, inverse_sigmoid_torch, Timer, sample_two_point, Camera, quat_multiply
from dataclasses import dataclass, field
import transforms as tf
import time
import math
import cv2
import numpy as np
from typing import Any, List
from einops import repeat
from renderer import draw, trunc_exp, global_culling, world2camera_func
from tqdm import tqdm
import argparse
from pykdtree.kdtree import KDTree
from plyfile import PlyData, PlyElement
import warnings
EPS=1e-4

def world_to_camera(points, rot, tran):
    warnings.warn("`world_to_camera` implementation using CUDA is known to be WRONG and is depracated! see: https://github.com/MultyXu/zero-splats/issues/1",
                  DeprecationWarning)
    # r = torch.empty_like(points)
    # gaussian.world2camera(points, rot, tran, r)
    # return r
    return world2camera_func(points, rot.transpose(-1,-2), tran)
    # _r = points @ rot.T + tran.unsqueeze(0)
    # return _r

def world_to_camera_corr(points: torch.Tensor, rot: torch.Tensor, tran: torch.Tensor) -> torch.Tensor:
    """
    - Inputs:
        - points: (...,dim) tensor
        - rot: (dim,dim) tensor
        - tran: (dim,) tensor
    """
    return (rot @ points[...,None]).squeeze(-1) + tran

def camera_to_image(points_camera_space):
    points_image_space = [
        points_camera_space[:,0]/points_camera_space[:,2],
        points_camera_space[:,1]/points_camera_space[:,2],
        points_camera_space.norm(dim=-1),
    ]
    return torch.stack(points_image_space, dim=-1)


#TODO ensure if the camera coordinate system is normalized
class Gaussian3ds(nn.Module):
    def __init__(self, pos, rgb, opa, quat=None, scale=None, cov=None, init_values=False):
        super().__init__()
        self.init_values = init_values
        if init_values:
            self.pos = nn.parameter.Parameter(pos)
            self.rgb = nn.parameter.Parameter(rgb)
            self.opa = nn.parameter.Parameter(opa)
            self.quat = quat if quat is None else nn.parameter.Parameter(quat)
            self.scale = scale if scale is None else nn.parameter.Parameter(scale)
            self.cov = cov if cov is None else nn.parameter.Parameter(cov)
        else:
            self.pos = pos
            self.rgb = rgb
            self.opa = opa
            self.quat = quat
            self.scale = scale
            self.cov = cov

    def to_scale_matrix(self):
        return torch.diag_embed(self.scale)

    def _tocpp(self):
        _cobj = gaussian.Gaussian3ds()
        _cobj.pos = self.pos.clone()
        _cobj.rgb = self.rgb.clone()
        _cobj.opa = self.opa.clone()
        _cobj.cov = self.cov.clone()
        return _cobj

    def to(self, *args, **kwargs):
        self.pos.to(*args, **kwargs)
        self.rgb.to(*args, **kwargs)
        self.opa.to(*args, **kwargs)
        if self.quat is not None:
            self.quat.to(*args, **kwargs)
        if self.scale is not None:
            self.scale.to(*args, **kwargs)
        if self.cov is not None:
            self.cov.to(*args, **kwargs)

    def filte(self, mask):
        if self.quat is not None and self.scale is not None:
            assert self.cov is None
            return Gaussian3ds(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opa=self.opa[mask],
                quat=self.quat[mask],
                scale=self.scale[mask],
            )
        else:
            assert self.cov is not None
            return Gaussian3ds(
                pos=self.pos[mask],
                rgb=self.rgb[mask],
                opa=self.opa[mask],
                cov=self.cov[mask],
            )

    # Deprecated, now fused to CUDA kernel in gaussian.global culling
    def get_gaussian_3d_cov(self, scale_activation="abs"):
        R = q2r(self.quat) # this function is doing normalization for the quaternion
        if scale_activation == "abs":
            _scale = self.scale.abs()+EPS
        elif scale_activation == "exp":
            _scale = trunc_exp(self.scale)
        else:
            print("No support scale activation")
            exit()
        # _scale = trunc_exp(self.scale)
        # _scale = torch.clamp(_scale, min=1e-4, max=0.1)
        S = torch.diag_embed(_scale)
        RS = torch.bmm(R, S)
        RSSR = torch.bmm(RS, RS.permute(0,2,1))
        return RSSR

    # def stable_cov(self):
        # return self.cov + 1e-2*torch.eye(2).unsqueeze(dim=0).to(self.cov)

    def reset_opa(self):
        '''
        Poved that opacity are stored in inverse_sigmoid space
        '''
        torch.nn.init.uniform_(self.opa, a=inverse_sigmoid(0.01), b=inverse_sigmoid(0.01))

    def adaptive_control(
        self,
        grad,
        taus,
        delete_thresh,
        scale_activation="abs",
        grad_thresh=0.0002,
        grad_aggregation="max",
        use_clone=True,
        use_split=True,
        clone_dt=0.01,
    ):
        # grad: B x 3
        # densification
        # 1. delete gaussians with small opacities
        assert self.init_values # only for the initial gaussians
        # print(inverse_sigmoid(0.01))
        # print(self.opa.min())
        # print(self.opa.max())
        if scale_activation == "abs":
            _mask = (self.opa > inverse_sigmoid(0.02)) & (self.scale.norm(dim=-1) < delete_thresh) #& (self.scale.abs().max(dim=-1)[0] > 5e-4)
        elif scale_activation == "exp":
            _mask = (self.opa > inverse_sigmoid(0.02)) & (self.scale.exp().norm(dim=-1) < delete_thresh) #& (self.scale.exp().max(dim=-1)[0] > 1e-4)
        else:
            print("Wrong activation")
            exit()

        self.pos = nn.parameter.Parameter(self.pos.detach()[_mask])
        self.rgb = nn.parameter.Parameter(self.rgb.detach()[_mask])
        self.opa = nn.parameter.Parameter(self.opa.detach()[_mask])
        self.quat = nn.parameter.Parameter(self.quat.detach()[_mask])
        self.scale = nn.parameter.Parameter(self.scale.detach()[_mask])
        grad = grad[_mask]
        print("DELETE: {} Gaussians".format((~_mask).sum()))
        # 2. clone or split

        if grad_aggregation == "max":
            densify_mask = grad.abs().max(-1)[0] > grad_thresh
        else:
            assert grad_aggregation == "mean"
            densify_mask = grad.abs().mean(-1) > grad_thresh

        cat_pos = [self.pos.clone().detach()]
        cat_rgb = [self.rgb.clone().detach()]
        cat_opa = [self.opa.clone().detach()]
        cat_quat = [self.quat.clone().detach()]
        cat_scale = [self.scale.clone().detach()]
        if densify_mask.any():
            scale_norm = self.scale.norm(dim=-1) if scale_activation == "abs" else self.scale.exp().norm(dim=-1)
            split_mask = scale_norm > taus
            clone_mask = scale_norm <= taus
            split_mask = split_mask & densify_mask
            clone_mask = clone_mask & densify_mask

            if clone_mask.any() and use_clone:
                cloned_pos = self.pos[clone_mask].clone().detach()
                cloned_pos -= grad[clone_mask] * clone_dt
                cloned_rgb = self.rgb[clone_mask].clone().detach()
                cloned_opa = self.opa[clone_mask].clone().detach()
                cloned_quat = self.quat[clone_mask].clone().detach()
                cloned_scale = self.scale[clone_mask].clone().detach()
                print("CLONE: {} Gaussians".format(cloned_pos.shape[0]))
                cat_pos.append(cloned_pos)
                cat_rgb.append(cloned_rgb)
                cat_opa.append(cloned_opa)
                cat_quat.append(cloned_quat)
                cat_scale.append(cloned_scale)

            if split_mask.any() and use_split:
                _scale = self.scale.clone().detach()
                if scale_activation == "abs":
                    _scale[split_mask] /= 1.6
                elif scale_activation == "exp":
                    _scale[split_mask] -= math.log(1.6)
                else:
                    print("Wrong activation")
                    exit()

                cat_scale[0] = _scale
                # cat_scale[0][split_mask] /= 1.6
                # self.scale = nn.parameter.Parameter(_scale)

                # sampling two positions
                this_cov = self.get_gaussian_3d_cov(scale_activation=scale_activation)[split_mask]
                p1, p2 = sample_two_point(self.pos[split_mask], this_cov)

                # split_pos = self.pos[split_mask].clone().detach()
                # split_pos -= grad[split_mask] * 0.01
                origin_pos = cat_pos[0]
                origin_pos[split_mask] = p1.detach()
                cat_pos[0] = origin_pos
                split_pos = p2.detach()
                split_rgb = self.rgb[split_mask].clone().detach()
                split_opa = self.opa[split_mask].clone().detach()
                split_quat = self.quat[split_mask].clone().detach()
                split_scale = _scale[split_mask].clone()
                print("SPLIT : {} Gaussians".format(split_pos.shape[0]))
                cat_pos.append(split_pos)
                cat_rgb.append(split_rgb)
                cat_opa.append(split_opa)
                cat_quat.append(split_quat)
                cat_scale.append(split_scale)
            self.pos = nn.parameter.Parameter(torch.cat(cat_pos))
            self.rgb = nn.parameter.Parameter(torch.cat(cat_rgb))
            self.opa = nn.parameter.Parameter(torch.cat(cat_opa))
            self.quat = nn.parameter.Parameter(torch.cat(cat_quat))
            self.scale = nn.parameter.Parameter(torch.cat(cat_scale))

    # Deprecated, now fused to CUDA kernel in gaussian.global culling
    def project(self, rot, tran, near, jacobian_calc, scale_activation="abs"):

        pos_cam_space = world_to_camera(self.pos, rot, tran)
        pos_img_space = camera_to_image(pos_cam_space)

        if jacobian_calc == "cuda":
            jacobian = torch.empty(pos_cam_space.shape[0], 3, 3, device=self.pos.device)
            gaussian.jacobian(pos_cam_space, jacobian)
        else:
            jacobian = jacobian_torch(pos_cam_space)
        gaussian_3d_cov = self.get_gaussian_3d_cov(scale_activation=scale_activation)
        # JW = torch.einsum("bij,bjk->bik", jacobian, rot.unsqueeze(dim=0))
        JW = torch.matmul(jacobian, rot.unsqueeze(dim=0))
        JWC = torch.bmm(JW, gaussian_3d_cov)
        gaussian_2d_cov = torch.bmm(JWC, JW.permute(0,2,1))[:, :2, :2]

        gaussian_3ds_image_space = Gaussian3ds(
            pos=pos_img_space,
            rgb=self.rgb.sigmoid(),
            opa=self.opa.sigmoid(),
            cov=gaussian_2d_cov,
        )
        return gaussian_3ds_image_space

class Tiles:
    def __init__(self, width, height, focal_x, focal_y, device):
        self.width = width
        self.height = height
        self.padded_width = int(math.ceil(self.width/16)) * 16
        self.padded_height = int(math.ceil(self.height/16)) * 16
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.n_tile_x = self.padded_width // 16
        self.n_tile_y = self.padded_height // 16
        self.device = device

    def crop(self, image):
        # image: padded_height x padded_width x 3
        # output: height x width x 3
        top = int(self.padded_height - self.height)//2
        left = int(self.padded_width - self.width)//2
        return image[top:top+int(self.height), left:left+int(self.width), :]

    def create_tiles(self):
        self.tiles_left = torch.linspace(-self.padded_width/2, self.padded_width/2, self.n_tile_x + 1, device=self.device)[:-1]
        self.tiles_right = self.tiles_left + 16
        self.tiles_top = torch.linspace(-self.padded_height/2, self.padded_height/2, self.n_tile_y + 1, device=self.device)[:-1]
        self.tiles_bottom = self.tiles_top + 16
        self.tile_geo_length_x = 16 / self.focal_x
        self.tile_geo_length_y = 16 / self.focal_y
        self.leftmost = -self.padded_width/2/self.focal_x
        self.topmost = -self.padded_height/2/self.focal_y

        self.tiles_left = self.tiles_left/self.focal_x
        self.tiles_top = self.tiles_top/self.focal_y
        self.tiles_right = self.tiles_right/self.focal_x
        self.tiles_bottom = self.tiles_bottom/self.focal_y

        self.tiles_left = repeat(self.tiles_left, "b -> (c b)", c=self.n_tile_y)
        self.tiles_right = repeat(self.tiles_right, "b -> (c b)", c=self.n_tile_y)

        self.tiles_top = repeat(self.tiles_top, "b -> (b c)", c=self.n_tile_x)
        self.tiles_bottom = repeat(self.tiles_bottom, "b -> (b c)", c=self.n_tile_x)

        _tile = gaussian.Tiles()
        _tile.top = self.tiles_top
        _tile.bottom = self.tiles_bottom
        _tile.left = self.tiles_left
        _tile.right = self.tiles_right
        return _tile

    def __len__(self):
        return self.tiles_top.shape[0]

class RayInfo:
    def __init__(self, w2c, tran, H, W, focal_x, focal_y):
        self.w2c = w2c
        self.c2w = torch.inverse(w2c)
        self.tran = tran
        self.H = H
        self.W = W
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.rays_o = - self.c2w @ tran

        lefttop_cam = torch.Tensor([(-W/2 + 0.5)/focal_x, (-H/2 + 0.5)/focal_y, 1.0]).to(self.w2c.device)
        dx_cam = torch.Tensor([1./focal_x, 0, 0]).to(self.w2c.device)
        dy_cam = torch.Tensor([0, 1./focal_y, 0]).to(self.w2c.device)
        self.lefttop = self.c2w @ (lefttop_cam - tran)
        self.dx = self.c2w @ dx_cam
        self.dy = self.c2w @ dy_cam

class Splatter(nn.Module):
    def __init__(self,
        colmap_path,
        image_path,
        near=0.3,
        #near=1.1,
        jacobian_calc="cuda",
        render_downsample=1,
        use_sh_coeff=False,
        render_weight_normalize=False,
        opa_init_value=0.1,
        scale_init_value=0.02,
        tile_culling_method="dist", # dist or prob
        tile_culling_dist_thresh=0.5,
        tile_culling_prob_thresh=0.1,
        debug=1,
        scale_activation="abs",
        cudaculling=0,
        load_ckpt=None,
        debug_align=False,
        fast_drawing=False,
        test=False,
    ):
        super().__init__()
        self.device = torch.device("cuda")
        self.use_sh_coeff = use_sh_coeff
        self.near = near
        self.jacobian_calc = jacobian_calc
        self.render_downsample = render_downsample
        self.render_weight_normalize = render_weight_normalize
        self.tile_culling_method = tile_culling_method
        self.tile_culling_dist_thresh = tile_culling_dist_thresh
        self.tile_culling_prob_thresh = tile_culling_prob_thresh
        self.debug = debug
        self.scale_activation = scale_activation
        self.cudaculling = cudaculling
        assert jacobian_calc == "cuda" or jacobian_calc == "torch"
        self.fast_drawing = fast_drawing

        self.points3d = read_points3d_binary(os.path.join(colmap_path, "points3D.bin"))
        self.cameras = read_cameras_binary(os.path.join(colmap_path,"cameras.bin"))
        self.images_info = read_images_binary(os.path.join(colmap_path,"images.bin"))
        self.image_path = image_path
        self.test = test
        if not self.test:
            self.parse_imgs()
        # self.vis_culling = Vis()
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc = torch.cuda.Event(enable_timing=True)

        _points = []
        _rgbs = []
        for pid, point in self.points3d.items():
            _points.append(torch.from_numpy(point.xyz))
            if self.use_sh_coeff:
                _rgbs.append(inverse_sigmoid_torch(torch.from_numpy(point.rgb/255.)))
            else:
                _rgbs.append(inverse_sigmoid_torch(torch.from_numpy(point.rgb/255.)))
            # _rgbs.append(torch.from_numpy(point.rgb))
            # self.vis_culling.add_item(point.id, point.xyz, point.rgb, point.error, point.image_ids, point.point2D_idxs)
        rgb = torch.stack(_rgbs).to(torch.float32).to(self.device) # B x 3
        if self.use_sh_coeff:
            rgb = initialize_sh(rgb)
            # rgb = torch.zeros(rgb.shape[0], 27).to(torch.float32).to(self.device)

        _pos=torch.stack(_points).to(torch.float32).to(self.device)
        if load_ckpt is None:
            _pos_np = _pos.cpu().numpy()
            kd_tree = KDTree(_pos_np)
            dist, idx = kd_tree.query(_pos_np, k=4)
            mean_min_three_dis = dist[:, 1:].mean(axis=1)
            mean_min_three_dis = torch.Tensor(mean_min_three_dis).to(torch.float32) * scale_init_value

            if scale_activation == "exp":
                mean_min_three_dis = mean_min_three_dis.log()

            self.gaussian_3ds = Gaussian3ds(
                pos=_pos.to(self.device), # B x 3
                rgb = rgb, # B x 3 or 27
                opa = torch.ones(len(_points)).to(torch.float32).to(self.device)*inverse_sigmoid(opa_init_value), # B
                quat = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4
                scale = torch.ones(len(_points), 3).to(torch.float32).to(self.device)*mean_min_three_dis.unsqueeze(dim=1).to(self.device),
                init_values=True,
            )
        else:
            self.gaussian_3ds = Gaussian3ds(
                pos=_pos.to(self.device), # B x 3
                rgb = rgb, # B x 3 or 27
                opa = torch.ones(len(_points)).to(torch.float32).to(self.device)*inverse_sigmoid(opa_init_value), # B
                quat = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(len(_points),1).to(torch.float32).to(self.device), # B x 4
                scale = torch.ones(len(_points), 3).to(torch.float32).to(self.device),
                init_values=True,
            )

        if load_ckpt is not None:
            # load checkpoint
            ckpt = torch.load(load_ckpt)
            self.gaussian_3ds.pos = nn.Parameter(ckpt["pos"])
            self.gaussian_3ds.opa = nn.Parameter(ckpt["opa"])
            self.gaussian_3ds.rgb = nn.Parameter(ckpt["rgb"])
            self.gaussian_3ds.quat = nn.Parameter(ckpt["quat"])
            self.gaussian_3ds.scale = nn.Parameter(ckpt["scale"])
        self.current_camera = None
        if not self.test:
            self.set_camera(0)

    def parse_imgs(self):
        img_ids = sorted([im.id for im in self.images_info.values()])
        self.w2c_quats = []
        self.w2c_rots = []
        self.w2c_trans = []
        self.cam_ids = []
        self.imgs = []
        for img_id in tqdm(img_ids):
            img_info = self.images_info[img_id]
            cam = self.cameras[img_info.camera_id]
            image_filename = os.path.join(self.image_path, img_info.name)
            if not os.path.exists(image_filename):
                continue
            _current_image = cv2.imread(image_filename)
            _current_image = cv2.cvtColor(_current_image, cv2.COLOR_BGR2RGB)
            self.imgs.append(torch.from_numpy(_current_image).to(torch.uint8).to(self.device))

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img_info.qvec), img_info.tvec,
            )#.inverse()
            self.w2c_quats.append(torch.from_numpy(T_world_camera.rotation().wxyz).to(torch.float32).to(self.device))
            self.w2c_trans.append(torch.from_numpy(T_world_camera.translation()).to(torch.float32).to(self.device))
            self.w2c_rots.append(q2r(self.w2c_quats[-1].unsqueeze(0)).squeeze().to(torch.float32).to(self.device))
            # print(self.w2c_trans)
            # print(self.w2c_rots)
            self.cam_ids.append(img_info.camera_id)

    def switch_resolution(self, downsample_factor):
        if downsample_factor == self.render_downsample:
            return
        self.image_path = self.image_path.replace(f"images_{self.render_downsample}", f"images_{downsample_factor}")
        self.render_downsample = downsample_factor
        self.parse_imgs()
        self.current_camera = None
        self.set_camera(0)
        # print(torch.stack(self.w2c_trans, dim=0).mean(0))
        # print(torch.stack(self.w2c_rots, dim=0).mean(0))

    def set_camera(self, idx, extrinsics=None, intrinsics=None):
        if idx is None:
            # print(extrinsics)
            self.current_w2c_rot = torch.from_numpy(extrinsics["rot"]).to(torch.float32).to(self.device)
            self.current_w2c_tran = torch.from_numpy(extrinsics["tran"]).to(torch.float32).to(self.device)
            self.current_w2c_quat = None
            self.ground_truth = None
            self.current_camera = Camera(
                id=-1, model="pinhole", width=intrinsics["width"], height=intrinsics["height"],
                params = np.array(
                    [intrinsics["focal_x"], intrinsics["focal_y"]]
                ),
            )
            self.tile_info = Tiles(
                math.ceil(intrinsics["width"]),
                math.ceil(intrinsics["height"]),
                intrinsics["focal_x"],
                intrinsics["focal_y"],
                self.device
            )
            self.tile_info_cpp = self.tile_info.create_tiles()
        else:
            with Timer("    set image", debug=self.debug):
                self.current_w2c_quat = self.w2c_quats[idx]
                self.current_w2c_tran = self.w2c_trans[idx]
                self.current_w2c_rot = self.w2c_rots[idx]
                self.ground_truth = self.imgs[idx].to(torch.float16)/255.
            with Timer("    set camera", debug=self.debug):
                if self.cameras[self.cam_ids[idx]] != self.current_camera:
                    self.current_camera = self.cameras[self.cam_ids[idx]]
                    width = self.current_camera.width / self.render_downsample
                    height = self.current_camera.height / self.render_downsample
                    focal_x = self.current_camera.params[0] / self.render_downsample
                    focal_y = self.current_camera.params[1] / self.render_downsample
                    self.tile_info = Tiles(int(self.ground_truth.shape[1]), int(self.ground_truth.shape[0]), focal_x, focal_y, self.device)
                    self.tile_info_cpp = self.tile_info.create_tiles()

        self.ray_info = RayInfo(
            w2c=self.current_w2c_rot,
            tran=self.current_w2c_tran,
            H=self.tile_info.padded_height,
            W=self.tile_info.padded_width,
            focal_x=self.tile_info.focal_x,
            focal_y=self.tile_info.focal_y
        )

    def project_and_culling(self):
        # project 3D to 2D
        # print(f"number of gaussians {len(self.gaussian_3ds.pos)}")
        # self.tic.record()
        if self.cudaculling:
            with Timer(" frustum cuda", debug=self.debug):
                normed_quat = (self.gaussian_3ds.quat/self.gaussian_3ds.quat.norm(dim=1, keepdim=True))
                if self.scale_activation == "abs":
                    normed_scale = self.gaussian_3ds.scale.abs()+EPS
                else:
                    assert self.scale_activation == "exp"
                    normed_scale = trunc_exp(self.gaussian_3ds.scale)
                _pos, _cov, _culling_mask = global_culling(
                    self.gaussian_3ds.pos,
                    normed_quat,
                    normed_scale,
                    self.current_w2c_rot.detach(),
                    self.current_w2c_tran.detach(),
                    self.near,
                    self.current_camera.width*1.2/2/self.current_camera.params[0],
                    self.current_camera.height*1.2/2/self.current_camera.params[1],
                )

                self.culling_gaussian_3d_image_space = Gaussian3ds(
                    pos=_pos[_culling_mask.bool()],
                    cov=_cov[_culling_mask.bool()],
                    rgb=self.gaussian_3ds.rgb[_culling_mask.bool()] if self.use_sh_coeff else self.gaussian_3ds.rgb[_culling_mask.bool()].sigmoid(),
                    opa=self.gaussian_3ds.opa[_culling_mask.bool()].sigmoid(),
                )
                self.culling_mask = _culling_mask
        else:
            with Timer("culling 1"):
                gaussian_3ds_pos_camera_space = world_to_camera(self.gaussian_3ds.pos, self.current_w2c_rot, self.current_w2c_tran)
            with Timer("culling 2"):
                valid = gaussian_3ds_pos_camera_space[:,2] > self.near
                gaussian_3ds_pos_image_space = camera_to_image(gaussian_3ds_pos_camera_space)
                culling_mask = (gaussian_3ds_pos_image_space[:, 0].abs() < (self.current_camera.width*1.2/2/self.current_camera.params[0]))  & \
                                (gaussian_3ds_pos_image_space[:, 1].abs() < (self.current_camera.height*1.2/2/self.current_camera.params[1]))
                valid &= culling_mask
                self.gaussian_3ds_valid = self.gaussian_3ds.filte(valid)
            with Timer("cullint 3"):
                self.culling_gaussian_3d_image_space = self.gaussian_3ds_valid.project(
                    self.current_w2c_rot,
                    self.current_w2c_tran,
                    self.near,
                    self.jacobian_calc,
                    scale_activation=self.scale_activation,
                )

    def render(self, out_write=True):
        if len(self.culling_gaussian_3d_image_space.pos) == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)
        # self.tic.record()
        with Timer("     culling tiles", debug=self.debug):
            tile_n_point = torch.zeros(len(self.tile_info), device=self.device, dtype=torch.int32)
            # MAXP = len(self.culling_gaussian_3d_image_space.pos)//10
            MAXP = len(self.culling_gaussian_3d_image_space.pos)//20
            tile_gaussian_list = torch.ones(len(self.tile_info), MAXP, device=self.device, dtype=torch.int32) * -1
            _method_config = {"dist": 0, "prob": 1, "prob2": 2}
            gaussian.calc_tile_list(
                    self.culling_gaussian_3d_image_space._tocpp(),
                    self.tile_info_cpp,
                    tile_n_point,
                    tile_gaussian_list,
                    (self.tile_info.tile_geo_length_x/self.tile_culling_dist_thresh) ** 2 if self.tile_culling_method == "dist" else self.tile_culling_prob_thresh,
                    _method_config[self.tile_culling_method],
                    self.tile_info.tile_geo_length_x,
                    self.tile_info.tile_geo_length_y,
                    self.tile_info.n_tile_x,
                    self.tile_info.n_tile_y,
                    self.tile_info.leftmost,
                    self.tile_info.topmost,
            )
            tile_n_point = torch.min(tile_n_point, torch.ones_like(tile_n_point)*MAXP)

        if tile_n_point.sum() == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)

        with Timer("     gather culled tiles", debug=self.debug):
            gathered_list = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
            tile_ids_for_points = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
            tile_n_point_accum = torch.cat([torch.Tensor([0]).to(self.device), torch.cumsum(tile_n_point, 0)]).to(tile_n_point)
            max_points_for_tile = tile_n_point.max().item()
            # print(max_points_for_tile)
            gaussian.gather_gaussians(
                tile_n_point_accum,
                tile_gaussian_list,
                gathered_list,
                tile_ids_for_points,
                int(max_points_for_tile),
            )
            self.tile_gaussians = self.culling_gaussian_3d_image_space.filte(gathered_list.long())
            self.n_tile_gaussians = len(self.tile_gaussians.pos)
            self.n_gaussians = len(self.gaussian_3ds.pos)

        with Timer("     sorting", debug=self.debug):
            # cat id and sort
            BASE = self.tile_gaussians.pos[..., 2].max()
            id_and_depth = self.tile_gaussians.pos[..., 2].to(torch.float32) + tile_ids_for_points.to(torch.float32) * (BASE+1)
            _, sort_indices = torch.sort(id_and_depth)
            self.tile_gaussians = self.tile_gaussians.filte(sort_indices)

        with Timer("     rendering", debug=self.debug):
            rendered_image = draw(
                self.tile_gaussians.pos,
                self.tile_gaussians.rgb,
                self.tile_gaussians.opa,
                self.tile_gaussians.cov,
                tile_n_point_accum,
                self.tile_info.padded_height,
                self.tile_info.padded_width,
                self.tile_info.focal_x,
                self.tile_info.focal_y,
                self.render_weight_normalize,
                False,
                self.use_sh_coeff,
                self.fast_drawing,
                self.ray_info.rays_o,
                self.ray_info.lefttop,
                self.ray_info.dx,
                self.ray_info.dy,
            )

        with Timer("    write out", debug=self.debug):
            if out_write:
                img_npy = rendered_image.clip(0,1).detach().cpu().numpy()
                cv2.imwrite("test.png", (img_npy*255).astype(np.uint8)[...,::-1])

        return rendered_image

    def forward(self, camera_id=None, extrinsics=None, intrinsics=None):
        with Timer("forward", debug=self.debug):
            with Timer("set camera", debug=self.debug):
                self.set_camera(camera_id, extrinsics, intrinsics)
            with Timer("frustum culling", debug=self.debug):
                self.project_and_culling()
            with Timer("render function", debug=self.debug):
                padded_render_img = self.render(out_write=False)
            with Timer("crop", debug=self.debug):
                padded_render_img = torch.clamp(padded_render_img, 0, 1)
                ret = self.tile_info.crop(padded_render_img)

        return ret


class StableSplatter(nn.Module):
    '''
    intended format
        xyz: real number space
        rgb: 0 ~ 1 ?? (TODO: don't know what's the correct storage format yet)
        scale: 0 ~ 1
        quat: normalized
        opacity: 0 ~ 1

    storage format
        xyz: real number space
        rgb: real number space space (activation: unkonw, seems to be related to spherical harmonics)
        scale: positive number space (activation exp)
        quat: unnormalized real number space (activation: normalzie)
        opacity: inverse-sigmoid space (orignal 0 ~ 1) (activation: sigmoid)

    '''
    def __init__(self,
        load_dict=None,
        load_ckpt=None,
        load_ply=None,
        load_network_output=None,
        load_tensor=None,
        near=0.3,
        #near=1.1,
        jacobian_calc="cuda",
        render_downsample=1,
        use_sh_coeff=False,
        render_weight_normalize=False,
        tile_culling_method="prob", # dist or prob
        tile_culling_dist_thresh=0.5,
        tile_culling_prob_thresh=0.1,
        debug=1,
        scale_activation="abs",
        cudaculling=0,
        debug_align=False,
        fast_drawing=False,
        test=False,
        init_values=False,
    ):
        super().__init__()
        assert load_dict is not None or load_ckpt is not None or load_ply is not None or load_network_output is not None or load_tensor is not None


        self.device = torch.device("cuda")
        self.use_sh_coeff = use_sh_coeff
        self.near = near
        self.jacobian_calc = jacobian_calc
        self.render_downsample = render_downsample
        self.render_weight_normalize = render_weight_normalize
        self.tile_culling_method = tile_culling_method
        self.tile_culling_dist_thresh = tile_culling_dist_thresh
        self.tile_culling_prob_thresh = tile_culling_prob_thresh
        self.debug = debug
        self.scale_activation = scale_activation
        self.cudaculling = cudaculling
        assert jacobian_calc == "cuda" or jacobian_calc == "torch"
        self.fast_drawing = fast_drawing

        # some activation
        # scale activation is usually exp
        # self.quat_activation = torch.nn.functional.normalize
        # self.opa_activation = torch.nn.functional.sigmoid

        # self.vis_culling = Vis()
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc = torch.cuda.Event(enable_timing=True)

        if load_dict is not None:
            self.load_from_dict(load_dict)
        elif load_ckpt is not None:
            # load checkpoint
            ckpt = torch.load(load_ckpt)
            self.load_from_dict(ckpt)
        elif load_ply is not None:
            self.load_from_ply(load_ply)
        elif load_network_output is not None:
            self.load_network_output(load_network_output)
        elif load_tensor is not None:
            self.load_from_tensor(load_tensor, init_values)
        else:
            raise NotImplementedError

        self.current_camera = None


    def update_gaussian_3ds(self, pos=None, rgb=None, opa=None, quat=None, scale=None):
        if pos is not None:
            self.gaussian_3ds.pos = pos
        if rgb is not None:
            self.gaussian_3ds.rgb = rgb
        if opa is not None:
            self.gaussian_3ds.opa = opa
        if quat is not None:
            self.gaussian_3ds.quat = quat
        if scale is not None:
            self.gaussian_3ds.scale = scale

    def load_from_tensor(self, tensor : torch.Tensor, init_values=False):
        '''
        Given tensor [N, 14]
        0:3 = xyz
        3:6 = rgb
        6:9 = scale
        9:13 = quaternion
        13 = opacity
        '''
        self.gaussian_3ds = Gaussian3ds(
                pos = tensor[:,0:3], # B x 3
                rgb = tensor[:,3:6], # B x 3 or 27
                scale = tensor[:,6:9], # B x 3
                quat = tensor[:,9:13], # B x 4
                opa = tensor[:,13].unsqueeze(1), # B x 1
                init_values=init_values,
            )


    def load_from_dict(self, ckpt, init_values=False):
        self.gaussian_3ds = Gaussian3ds(
                pos = ckpt["pos"], # B x 3
                rgb = ckpt["rgb"], # B x 3 or 27
                scale = ckpt["scale"],
                quat = ckpt["quat"], # B x 4
                opa = ckpt["opa"], # B
                init_values=init_values,
            )

        return


    def save_ckpt(self, path='ckpt.pth'):
        ckpt = {
            "pos": self.gaussian_3ds.pos,
            "rgb": self.gaussian_3ds.rgb,
            "scale": self.gaussian_3ds.scale,
            "quat": self.gaussian_3ds.quat,
            "opa": self.gaussian_3ds.opa,
        }
        torch.save(ckpt, path)


    def load_from_ply(self, path, init_values=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        # print(features_extra.shape,len(extra_f_names))
        # raise

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.gaussian_3ds = Gaussian3ds(
            pos = torch.tensor(xyz, dtype=torch.float, device=self.device), # B x 3
            rgb = torch.tensor(features_dc, dtype=torch.float, device=self.device), # B x 3 or 27
            opa = torch.tensor(opacities, dtype=torch.float, device=self.device), # B x 1
            quat = torch.tensor(rots, dtype=torch.float, device=self.device), # B x 4
            scale = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device)), # B x 3
            init_values=init_values,
        )
        return

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.gaussian_3ds.rgb.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.gaussian_3ds.scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.gaussian_3ds.quat.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path='ckpt.ply'):

        xyz = self.gaussian_3ds.pos.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.gaussian_3ds.rgb.detach().cpu().numpy()
        f_rest = torch.zeros(xyz.shape[0], 45).cpu().numpy()
        opacities = self.gaussian_3ds.opa.detach().cpu().numpy()
        # sotred in normal space but ply soted in log space
        scale = torch.log(self.gaussian_3ds.scale.detach()).cpu().numpy()
        rotation = self.gaussian_3ds.quat.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_network_output(self, nn_output):
        '''
        construct a gaussian model using the output from the neural network
        input: nn_output tensor, flatten featues with format [xyz, dc, s, r, a]
        '''
        unflattened = nn_output.reshape(14, -1)

        # self.gaussian_3ds = Gaussian3ds(
        #     pos = nn.Parameter(unflattened[0:3].T).to(self.device),
        #     rgb = nn.Parameter(unflattened[3:6].T).to(self.device),
        #     # TODO: need to figure out what is features_rest
        #     # self._features_rest = nn.Parameter(torch.zeros(unflattened.shape[1],15,3).contiguous()).to(nn_output.device)
        #     scale = nn.Parameter(unflattened[6:9].T).to(self.device),
        #     quat= nn.Parameter(unflattened[9:13].T).to(self.device),
        #     opa = nn.Parameter(unflattened[13].reshape(-1,1)).to(self.device),
        #     init_values=True
        # )

        self.gaussian_3ds = Gaussian3ds(
            pos = unflattened[0:3].T.to(self.device),
            rgb = unflattened[3:6].T.to(self.device),
            # TODO: need to figure out what is features_rest
            # self._features_rest = nn.Parameter(torch.zeros(unflattened.shape[1],15,3).contiguous()).to(nn_output.device)
            scale = unflattened[6:9].T.to(self.device),
            quat= unflattened[9:13].T.to(self.device),
            opa = unflattened[13].reshape(-1,1).to(self.device),
            init_values=True
        )

    def get_splat_parameters(self) -> torch.tensor:
        '''
        Return a tensor of (N,14)
        0:3 = xyz
        3:6 = rgb
        6:9 = scale
        9:13 = quaternion
        13 = opacity
        '''
        return torch.concat((
            self.gaussian_3ds.pos,
            self.gaussian_3ds.rgb,
            self.gaussian_3ds.scale,
            self.gaussian_3ds.quat,
            self.gaussian_3ds.opa),dim=1)

    def get_falttened_properties_for_training(self, num_splats = None):
        '''
        flatten all the feature and return it as a 1D vector
        '''
        if num_splats is None:
            # default flatten all features
            num_splats = self.gaussian_3ds.pos.shape[0]

        return ( torch.concat(
                (self.gaussian_3ds.pos[:num_splats].T.flatten(0), # [size, 3]
                self.gaussian_3ds.rgb[:num_splats].T.flatten(0), # [size, 3]
                self.gaussian_3ds.scale[:num_splats].T.flatten(0), # [size, 3]
                self.gaussian_3ds.quat[:num_splats].T.flatten(0), # [size, 4]
                self.gaussian_3ds.opa[:num_splats].T.flatten(0) # [size, 1]
                ),
                dim=0
            )
        )


    def rotate_splat(self, unit_quat):
        '''
        input: unit_quat (4,) torch.tensor of random quaternion
        '''
        unit_quat = unit_quat.to(self.device)
        t_to_0 = torch.mean(self.gaussian_3ds.pos, dim=0).reshape(-1,3)
        center_pos = self.gaussian_3ds.pos - t_to_0
        R = q2r(unit_quat.unsqueeze(0)) # B x 3 x 3, B = 1 here
        rotated_pos = torch.matmul(R[0], center_pos[...,None]).squeeze(-1) # n x 3 new pos
        # print(rotated_pos.shape)
        rotated_pos = rotated_pos + t_to_0 # give back the center offset
        normalized_quat = self.gaussian_3ds.quat / torch.norm(self.gaussian_3ds.quat, dim=-1, keepdim=True)
        rotated_quat = quat_multiply(normalized_quat, unit_quat)

        # update the splat
        self.gaussian_3ds.pos = rotated_pos.to(self.device)
        self.gaussian_3ds.quat = rotated_quat.to(self.device)


    def set_camera(self, extrinsics=None, intrinsics=None):

        # print(extrinsics)
        self.current_w2c_rot = torch.from_numpy(extrinsics["rot"]).to(torch.float32).to(self.device)
        self.current_w2c_tran = torch.from_numpy(extrinsics["tran"]).to(torch.float32).to(self.device)
        self.current_w2c_quat = None
        self.ground_truth = None
        self.current_camera = Camera(
            id=-1, model="pinhole", width=intrinsics["width"], height=intrinsics["height"],
            params = np.array(
                [intrinsics["focal_x"], intrinsics["focal_y"]]
            ),
        )
        self.tile_info = Tiles(
            math.ceil(intrinsics["width"]),
            math.ceil(intrinsics["height"]),
            intrinsics["focal_x"],
            intrinsics["focal_y"],
            self.device
        )
        self.tile_info_cpp = self.tile_info.create_tiles()

        self.ray_info = RayInfo(
            w2c=self.current_w2c_rot,
            tran=self.current_w2c_tran,
            H=self.tile_info.padded_height,
            W=self.tile_info.padded_width,
            focal_x=self.tile_info.focal_x,
            focal_y=self.tile_info.focal_y
        )

    def project_and_culling(self):
        # project 3D to 2D
        # print(f"number of gaussians {len(self.gaussian_3ds.pos)}")
        # self.tic.record()
        if self.cudaculling:
            with Timer(" frustum cuda", debug=self.debug):
                normed_quat = (self.gaussian_3ds.quat/self.gaussian_3ds.quat.norm(dim=1, keepdim=True))
                if self.scale_activation == "abs":
                    normed_scale = self.gaussian_3ds.scale.abs()+EPS
                else:
                    assert self.scale_activation == "exp"
                    normed_scale = trunc_exp(self.gaussian_3ds.scale)
                _pos, _cov, _culling_mask = global_culling(
                    self.gaussian_3ds.pos,
                    normed_quat,
                    normed_scale,
                    self.current_w2c_rot.detach(),
                    self.current_w2c_tran.detach(),
                    self.near,
                    self.current_camera.width*1.2/2/self.current_camera.params[0],
                    self.current_camera.height*1.2/2/self.current_camera.params[1],
                )

                self.culling_gaussian_3d_image_space = Gaussian3ds(
                    pos=_pos[_culling_mask.bool()],
                    cov=_cov[_culling_mask.bool()],
                    rgb=self.gaussian_3ds.rgb[_culling_mask.bool()] if self.use_sh_coeff else self.gaussian_3ds.rgb[_culling_mask.bool()].sigmoid(),
                    opa=self.gaussian_3ds.opa[_culling_mask.bool()].sigmoid(),
                )
                self.culling_mask = _culling_mask
        else:
            with Timer("culling 1"):
                gaussian_3ds_pos_camera_space = world_to_camera_corr(self.gaussian_3ds.pos, self.current_w2c_rot, self.current_w2c_tran)
            with Timer("culling 2"):
                valid = gaussian_3ds_pos_camera_space[:,2] > self.near
                gaussian_3ds_pos_image_space = camera_to_image(gaussian_3ds_pos_camera_space)
                culling_mask = (gaussian_3ds_pos_image_space[:, 0].abs() < (self.current_camera.width*1.2/2/self.current_camera.params[0]))  & \
                                (gaussian_3ds_pos_image_space[:, 1].abs() < (self.current_camera.height*1.2/2/self.current_camera.params[1]))
                valid &= culling_mask
                self.gaussian_3ds_valid = self.gaussian_3ds.filte(valid)
            with Timer("cullint 3"):
                self.culling_gaussian_3d_image_space = self.gaussian_3ds_valid.project(
                    self.current_w2c_rot,
                    self.current_w2c_tran,
                    self.near,
                    self.jacobian_calc,
                    scale_activation=self.scale_activation,
                )

    def render(self, out_write=True):
        if len(self.culling_gaussian_3d_image_space.pos) == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)
        # self.tic.record()
        with Timer("     culling tiles", debug=self.debug):
            tile_n_point = torch.zeros(len(self.tile_info), device=self.device, dtype=torch.int32)
            # MAXP = len(self.culling_gaussian_3d_image_space.pos)//10
            MAXP = len(self.culling_gaussian_3d_image_space.pos)//20
            tile_gaussian_list = torch.ones(len(self.tile_info), MAXP, device=self.device, dtype=torch.int32) * -1
            _method_config = {"dist": 0, "prob": 1, "prob2": 2}
            gaussian.calc_tile_list(
                    self.culling_gaussian_3d_image_space._tocpp(),
                    self.tile_info_cpp,
                    tile_n_point,
                    tile_gaussian_list,
                    (self.tile_info.tile_geo_length_x/self.tile_culling_dist_thresh) ** 2 if self.tile_culling_method == "dist" else self.tile_culling_prob_thresh,
                    _method_config[self.tile_culling_method],
                    self.tile_info.tile_geo_length_x,
                    self.tile_info.tile_geo_length_y,
                    self.tile_info.n_tile_x,
                    self.tile_info.n_tile_y,
                    self.tile_info.leftmost,
                    self.tile_info.topmost,
            )
            tile_n_point = torch.min(tile_n_point, torch.ones_like(tile_n_point)*MAXP)

        if tile_n_point.sum() == 0:
            return torch.zeros(self.tile_info.padded_height, self.tile_info.padded_width, 3, device=self.device, dtype=torch.float32)

        with Timer("     gather culled tiles", debug=self.debug):
            gathered_list = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
            tile_ids_for_points = torch.empty(tile_n_point.sum(), dtype=torch.int32, device=self.device)
            tile_n_point_accum = torch.cat([torch.Tensor([0]).to(self.device), torch.cumsum(tile_n_point, 0)]).to(tile_n_point)
            max_points_for_tile = tile_n_point.max().item()
            # print(max_points_for_tile)
            gaussian.gather_gaussians(
                tile_n_point_accum,
                tile_gaussian_list,
                gathered_list,
                tile_ids_for_points,
                int(max_points_for_tile),
            )
            self.tile_gaussians = self.culling_gaussian_3d_image_space.filte(gathered_list.long())
            self.n_tile_gaussians = len(self.tile_gaussians.pos)
            self.n_gaussians = len(self.gaussian_3ds.pos)

        with Timer("     sorting", debug=self.debug):
            # cat id and sort
            BASE = self.tile_gaussians.pos[..., 2].max()
            id_and_depth = self.tile_gaussians.pos[..., 2].to(torch.float32) + tile_ids_for_points.to(torch.float32) * (BASE+1)
            _, sort_indices = torch.sort(id_and_depth)
            self.tile_gaussians = self.tile_gaussians.filte(sort_indices)

        with Timer("     rendering", debug=self.debug):
            rendered_image = draw(
                self.tile_gaussians.pos,
                self.tile_gaussians.rgb,
                self.tile_gaussians.opa,
                self.tile_gaussians.cov,
                tile_n_point_accum,
                self.tile_info.padded_height,
                self.tile_info.padded_width,
                self.tile_info.focal_x,
                self.tile_info.focal_y,
                self.render_weight_normalize,
                False,
                self.use_sh_coeff,
                self.fast_drawing,
                self.ray_info.rays_o,
                self.ray_info.lefttop,
                self.ray_info.dx,
                self.ray_info.dy,
            )

        with Timer("    write out", debug=self.debug):
            if out_write:
                img_npy = rendered_image.clip(0,1).detach().cpu().numpy()
                cv2.imwrite("test.png", (img_npy*255).astype(np.uint8)[...,::-1])

        return rendered_image

    def forward(self, extrinsics=None, intrinsics=None):
        with Timer("forward", debug=self.debug):
            with Timer("set camera", debug=self.debug):
                self.set_camera(extrinsics, intrinsics)
            with Timer("frustum culling", debug=self.debug):
                self.project_and_culling()
            with Timer("render function", debug=self.debug):
                padded_render_img = self.render(out_write=False)
            with Timer("crop", debug=self.debug):
                padded_render_img = torch.clamp(padded_render_img, 0, 1)
                ret = self.tile_info.crop(padded_render_img)

        return ret


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cudaculling", type=int, default=0)
    # opt = parser.parse_args()
    # test = Splatter(
    #     os.path.join("colmap_garden/sparse/0/"),
    #     "colmap_garden/images_4/",
    #     render_weight_normalize=False,
    #     jacobian_calc="cuda",
    #     render_downsample=4,
    #     opa_init_value=0.8,
    #     scale_init_value=0.2,
    #     cudaculling=opt.cudaculling,
    #     load_ckpt="ckpt.pth",
    #     scale_activation="exp",
    # )
    # test.forward(camera_id=0)
    # loss = (test.ground_truth - test.forward(camera_id=0)).abs().mean()
    # loss.backward()

    random_params = torch.randn(1000, 14)
    splat = StableSplatter(load_tensor=random_params)
    from_splat = splat.get_splat_parameters()

    print(random_params == from_splat)