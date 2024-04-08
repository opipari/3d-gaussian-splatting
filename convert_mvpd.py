
import open3d as o3d
from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_cameras
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import matrix_to_quaternion

import struct
import torch
import numpy as np

root = '/home/topipari/Desktop/FastSPAM/video_segmentation/datasets/MVPd/MVPd/'
split = 'test'
dataset = MVPDataset(root=root,
						split=split,
						window_size = 0)
print(len(dataset))


camera_extrinsics_ = []
camera_extrinsics = []
camera_intrinsics = []
point_positions = []
point_colors = []
image_names = []
for video in dataset:
	if video.video_meta['video_name']!='00808-y9hTuugGdiq.0000000000.0000001000':
		continue

	PyTorch3dViewCoSys2Camera = torch.tensor([[[-1.0,  0.0,  0.0,  0.0],
								                 [ 0.0,  -1.0,  0.0,  0.0],
								                 [ 0.0,  0.0, 1.0,  0.0],
								                 [ 0.0,  0.0,  0.0,  1.0]]])
	Camera2PyTorch3dViewCoSys = PyTorch3dViewCoSys2Camera

	for idx, sample in enumerate(video):
		# Load metadata
		video_name = sample['meta']['video_name']
		
		image = torch.tensor(sample['observation']['image'][0], dtype=torch.uint8).reshape(1,-1,3) # 480 x 640 x 3

		# image = torch.tensor(sample['observation']['image']).permute(0,3,1,2).to('cuda')
		depth = torch.tensor(sample['observation']['depth']).unsqueeze(1).to('cuda')
		camera = get_cameras(sample['camera']['K'],
							sample['camera']['W2V_pose'],
							sample['meta']['image_size']).to('cuda')
		
		xy_depth = get_xy_depth(depth, from_ndc=True).permute(0,2,3,1).reshape(1,-1,3)
		xyz = camera.unproject_points(xy_depth, from_ndc=True, world_coordinates=True)

		rand_point_ind = torch.randperm(xyz.size(1))[:int(xyz.size(1)*0.005)]
		point_positions.append(xyz.squeeze(0)[rand_point_ind])
		point_colors.append(image.squeeze(0)[rand_point_ind])


		C2W = torch.matmul(Camera2PyTorch3dViewCoSys, torch.tensor(sample['camera']['V2W_pose']))
		W2C = torch.matmul(torch.tensor(sample['camera']['W2V_pose']), PyTorch3dViewCoSys2Camera)
		camera_extrinsics_.append(C2W)
		camera_extrinsics.append(W2C)
		camera_intrinsics.append(torch.tensor(sample['camera']['K'])[:,:3,:3])
		image_names.append(video_name+'/'+sample['meta']['window_names'][0])
		
point_positions = torch.cat(point_positions, dim=0)
point_colors = torch.cat(point_colors, dim=0)

camera_extrinsics_ = torch.cat(camera_extrinsics_, dim=0)
camera_extrinsics = torch.cat(camera_extrinsics, dim=0)
camera_intrinsics = torch.cat(camera_intrinsics, dim=0)
camera_intrinsics[:,2,2] = 1


def write_next_bytes(fid, format_char_sequence, *data_args, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = struct.pack(endian_character + format_char_sequence, *data_args)
    return fid.write(data)


def write_points3d_binary(path_to_model_file, 
						points_xyz: torch.Tensor, 
						points_rgb: torch.Tensor
						):
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(
                fid, "Q", points_xyz.shape[0]
            )
        for point_id, (point_xyz, point_rgb) in enumerate(zip(points_xyz, points_rgb)):
            write_next_bytes(
                fid, "Q", point_id
            )
            write_next_bytes(
                fid, "ddd", point_xyz[0].item(), point_xyz[1].item(), point_xyz[2].item()
            )
            write_next_bytes(
                fid, "BBB", point_rgb[0].item(), point_rgb[1].item(), point_rgb[2].item() 
            )
            write_next_bytes(
                fid, "d", 0 # error
            )
            write_next_bytes(
                fid, "Q", 0 # track length
            )


def write_cameras_binary(path_to_model_file, camera_intrinsics):
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, "Q", camera_intrinsics.shape[0])
        for camera_id, k in enumerate(camera_intrinsics):
            write_next_bytes(
                fid, "i", camera_id
            )
            write_next_bytes(
                fid, "i", 1 # model_id pinhole
            )
            write_next_bytes(
                fid, "Q", 640 # width
            )
            write_next_bytes(
                fid, "Q", 480 # height
            )

            write_next_bytes(
                fid, "dddd", k[0,0].item(), k[1,1].item(), k[0,2].item(), k[1,2].item() # fx fy cx cy
            )


def write_images_binary(path_to_model_file, image_qvecs, image_tvecs, image_names):
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, "Q", image_qvecs.shape[0])
        for image_id, (image_qvec, image_tvec, image_name) in enumerate(zip(image_qvecs, image_tvecs, image_names)):
            write_next_bytes(
                fid, "i", image_id
            )
            write_next_bytes(
                fid, "dddd", image_qvec[0].item(), image_qvec[1].item(), image_qvec[2].item(), image_qvec[3].item()
            )
            write_next_bytes(
                fid, "ddd", image_tvec[0].item(), image_tvec[1].item(), image_tvec[2].item()
            )
            write_next_bytes(
                fid, "i", image_id # camera_id
            )
            fid.write(bytes(image_name, 'utf-8')+b"\x00")
            # write_next_bytes(
            #     fid, "c"*len(image_name), bytes(image_name, 'utf-8') # camera_id
            # )

            write_next_bytes(
                fid, "Q", 0 # num_points2D
            )
            
            


# write_points3d_binary('mvpd_example/sparse/0/points3D.bin', point_positions, point_colors)
# write_cameras_binary('mvpd_example/sparse/0/cameras.bin', camera_intrinsics)
# write_images_binary('mvpd_example/sparse/0/images.bin', matrix_to_quaternion(torch.transpose(camera_extrinsics[:,:3,:3],1,2)), camera_extrinsics[:,3,:], image_names)






pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_positions.cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(point_colors.cpu().numpy()/255.0)


c_lines = []
for c,k in zip(camera_extrinsics, camera_intrinsics):
	c_lines.append(o3d.geometry.LineSet.create_camera_visualization(view_width_px=640, view_height_px=480, intrinsic=k.cpu().numpy(), extrinsic=c.T.cpu().numpy()))

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=[0, 0, 0])

cam_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=[0, 0, 0])

# print(cam_mesh_frame.pose)
# cam_mesh_frame.rotate(torch.transpose(camera_extrinsics[:,:3,:3],1,2)[0])
camera_extrinsics_ = torch.transpose(camera_extrinsics_,1,2)
# camera_extrinsics[:,:3,:3] = camera_extrinsics_[:,:3,:3]#torch.transpose(camera_extrinsics[:,:3,:3],1,2)
# camera_extrinsics[:,:3,3] = 0#-camera_extrinsics[:,:3,3]
cam_mesh_frame.transform(camera_extrinsics_[0])

print(camera_extrinsics_[0])
o3d.visualization.draw_geometries([mesh_frame,pcd,cam_mesh_frame]+c_lines)
