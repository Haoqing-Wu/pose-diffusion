import json
import os
import cv2
import torch
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree
from typing import Optional
from scipy.spatial.transform import Rotation
from posediff.modules.backbone.rotation_tools import compute_rotation_matrix_from_ortho6d




def sample_point_from_mesh(model_root,samples):
    r"""Sample given number of points from a mesh readed from path.
    """
    mesh = o3d.io.read_triangle_mesh(model_root)
    pcd = mesh.sample_points_uniformly(number_of_points=samples)
    scale_factor = 0.001
    pcd.scale(scale_factor,(0, 0, 0))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals

def get_bbox(bbox, img_size='linemod'):
    r"""Get bounding box from a mask.
    Return coordinates of the bounding box [x_min, y_min, x_max, y_max]
    """
    if img_size == 'linemod':
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        x_max = 640
        y_max = 480
    if img_size == 'tless':
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
                       720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]
        x_max = 1280
        y_max = 1024
    rmin, rmax, cmin, cmax = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
    rmin = max(rmin, 0)
    rmax = min(rmax, y_max-1)
    cmin = max(cmin, 0)
    cmax = min(cmax, x_max-1)
    r_b = rmax - rmin
    c_b = cmax - cmin

    for i in range(len(border_list) - 1):
        if r_b > border_list[i] and r_b < border_list[i + 1]:
            r_b = border_list[i + 1]
            break
    for i in range(len(border_list) - 1):
        if c_b > border_list[i] and c_b < border_list[i + 1]:
            c_b = border_list[i + 1]
            break

    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - max(int(r_b / 2), int(c_b / 2))
    rmax = center[0] + max(int(r_b / 2), int(c_b / 2))
    cmin = center[1] - max(int(r_b / 2), int(c_b / 2))
    cmax = center[1] + max(int(r_b / 2), int(c_b / 2))
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, y_max)
    cmax = min(cmax, x_max)

    return rmin, rmax, cmin, cmax


def mask_to_bbox(mask):
    r"""Get bounding box from a mask.
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return list(bbox)

def get_gt(gt_file, frame_id):
    r"""Get ground truth pose from a ground truth file.
    Return rotation matrix and translation vector
    """
    with open(gt_file, 'r') as file:
        gt = json.load(file)[str(frame_id)][0]
    rot = np.array(gt['cam_R_m2c']).reshape(3, 3)
    trans = np.array(gt['cam_t_m2c']) / 1000
    return rot, trans

def get_gt_scene(gt_file, frame_id):
    r"""Get ground truth pose from a ground truth file.
    Return rotation matrix and translation vector
    """
    gt_scene = []
    with open(gt_file, 'r') as file:
        gt_list = json.load(file)[str(frame_id)]
    for gt in gt_list:
        obj_id = gt['obj_id']
        rot = np.array(gt['cam_R_m2c']).reshape(3, 3)
        trans = np.array(gt['cam_t_m2c']) / 1000
        gt_scene.append({'obj_id': obj_id, 'rot': rot, 'trans': trans})
    return gt_scene

def get_camera_info(cam_file, frame_id):
    r"""Get camera intrinsics from a camera file.
    Return camera center, focal length
    """
    with open(cam_file, 'r') as file:
        cam = json.load(file)[str(frame_id)]
    cam_k = np.array(cam['cam_K']).reshape(3, 3)
    cam_cx = cam_k[0, 2]
    cam_cy = cam_k[1, 2]
    cam_fx = cam_k[0, 0]
    cam_fy = cam_k[1, 1]
    return cam_cx, cam_cy, cam_fx, cam_fy

def get_detection_file_length(detection_file):

    with open(detection_file, 'r') as file:
        detection_list = json.load(file)
    return len(detection_list)

def get_detection_meta(detection_file, idx):
    with open(detection_file, 'r') as file:
        detection_list = json.load(file)
    detection = detection_list[idx]
    scene_id = detection['scene_id']
    frame_id = detection['image_id']
    obj_id = detection['category_id']
    score = detection['score']
    bbox = detection['bbox']
    seg = detection['segmentation']
    time = detection['time']
    return {
        'scene_id': scene_id,
        'frame_id': frame_id,
        'obj_id': obj_id,
        'score': score,
        'bbox': bbox,
        'seg': seg,
        'time': time
    }
    

def get_model_symmetry(info_file, model_id):
    rot = np.eye(3)
    trans = np.zeros(3)
    with open(info_file, 'r') as file:
        model_info = json.load(file)[str(model_id)]
    if model_info.get('symmetries_continuous') is not None:
        symmetry = model_info['symmetries_continuous'][0]
        axis = np.array(symmetry['axis'])
        # randomize an angle
        angle = np.random.uniform(0, 2 * np.pi)
        rot = rotation_matrix_from_axis(axis, angle)
        trans = np.zeros(3)
    if model_info.get('symmetries_discrete') is not None:
        symmetries = model_info['symmetries_discrete']
        symmetry = symmetries[np.random.randint(0, len(symmetries))]
        symmetry = np.array(symmetry).reshape(4, 4)
        rot = symmetry[:3, :3]
        trans = symmetry[:3, 3] / 1000.0
        # not change if 0.5 probability
        if np.random.uniform(0., 1.) < 0.5:
            rot = np.eye(3)
            trans = np.zeros(3)

    return rot, trans

def rotation_matrix_from_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)

    sin_theta = np.sin(angle)
    cos_theta = np.cos(angle)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    return R

def resize_pcd(pcd, points_limit):
    r"""Resize a point cloud to a given number of points.
    """
    if pcd.shape[0] > points_limit:
        idx = np.random.permutation(pcd.shape[0])[:points_limit]
        pcd = pcd[idx]
    return pcd

def sort_pcd_from_center(pcd):
    r"""Sort a point cloud from the center.
    """
    center = np.mean(pcd, axis=0)
    pcd_v = pcd - center
    dist = np.sqrt(np.sum(np.square(pcd_v), axis=1))
    idx = np.argsort(dist)
    pcd = pcd[idx]
    return pcd

def transformation_pcd(pcd, rot, trans):
    r"""Transform a point cloud with a rotation matrix and a translation vector.
    """
    pcd_t = np.dot(pcd, rot.T)
    pcd_t = np.add(pcd_t, trans.T)
    return pcd_t

def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points

def get_transformation_inv(rot, trans):
    rot_inv = np.linalg.inv(rot)
    trans_inv = -np.matmul(rot_inv, trans)
    return rot_inv, trans_inv

def get_transformation_indirect(rot1, trans1, rot2, trans2):
    rot1_inv = np.linalg.inv(rot1)
    trans1_inv = -np.matmul(rot1_inv, trans1)

    transform1_inv = np.hstack((rot1_inv, trans1_inv.reshape(3, 1)))
    transform1_inv = np.vstack((transform1_inv, np.array([[0, 0, 0, 1]])))
    transform2 = np.hstack((rot2, trans2.reshape(3, 1)))
    transform2 = np.vstack((transform2, np.array([[0, 0, 0, 1]])))

    transform = np.matmul(transform2, transform1_inv)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return rot, trans


def get_corr(tgt_pcd, src_pcd, rot, trans, radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_t = transformation_pcd(src_pcd, rot, trans)
    src_tree = cKDTree(src_t)
    indices_list = src_tree.query_ball_point(tgt_pcd, radius)
    corr = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.int32,
    )
    coverage = corr.shape[0] / tgt_pcd.shape[0]
    return corr, coverage

def get_corr_score_matrix(tgt_pcd, src_pcd, transform, sigma=0.3):
    src_pcd_t = apply_transform(src_pcd, transform)
    # Calculate pointwise distances using broadcasting
    distances = torch.cdist(tgt_pcd, src_pcd_t)

    # Calculate pointwise scores using the Gaussian kernel
    scores = torch.exp(-(distances ** 2) / (2 * sigma ** 2))

    # Normalize scores to the range [-1, 1]
    scores = scores * 2 - 1

    return scores

def get_corr_matrix(corr, tgt_len, src_len):
    r"""Get a correspondence matrix from a correspondence array.
    Return correspondence matrix [tgt_len, src_len]
    """
    corr_matrix = np.full((tgt_len, src_len), -1.0, dtype=np.float32)
    corr_matrix[corr[:, 0], corr[:, 1]] = 1.0
    return corr_matrix

def get_corr_src_pcd(corr, src_pcd):
    r"""Get the source point cloud of the correspondences.
    Return source point cloud of the correspondences
    """
    return src_pcd[corr[:, 1]]

def get_corr_similarity(corr_matrix, gt_corr_matrix):
    r"""Get the cosine distance similarity between the correspondence matrix 
    and the ground truth correspondence matrix.
    Return cosine distance similarity
    """
    corr_matrix = corr_matrix.astype(np.float32)
    gt_corr_matrix = gt_corr_matrix.astype(np.float32)
    num = np.dot(corr_matrix, gt_corr_matrix.T)  
    denom = np.linalg.norm(corr_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(gt_corr_matrix, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def normalize_points(src, tgt, rot, trans):
    r"""Normalize point cloud to a unit sphere at origin."""

    src_factor = np.max(np.linalg.norm(src, axis=1))
    src = src / src_factor
    
    inv_rot = rot.T
    inv_trans = -np.matmul(inv_rot, trans)
    tgt = transformation_pcd(tgt, inv_rot, inv_trans)
    tgt = tgt / src_factor
    tgt = transformation_pcd(tgt, rot, trans)

    return src, tgt

def get_corr_from_matrix_topk(corr_matrix, k):
    r"""Get the top k correspondences from a correspondence matrix.[batch_size, tgt_len, src_len]
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    corr_indices = corr_matrix.view(-1).topk(k=k, largest=True)[1]
    ref_corr_indices = corr_indices // corr_matrix.shape[2]
    src_corr_indices = corr_indices % corr_matrix.shape[2]
    corr = np.array(
        [(i, j) for i, j in zip(ref_corr_indices, src_corr_indices)],
        dtype=np.int32,
    )
    return corr

def get_corr_from_matrix_gt(corr_matrix, low, high):
    r"""Get the between threshold correspondences from a correspondence matrix.[batch_size, tgt_len, src_len]
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    
    corr_indices = np.where((corr_matrix >= low))
    ref_corr_indices = corr_indices[1]
    src_corr_indices = corr_indices[2]
    corr = np.array(
        [(i, j) for i, j in zip(ref_corr_indices, src_corr_indices)],
        dtype=np.int32,
    )
    return corr, len(ref_corr_indices)

def isotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ['mean', 'sum', 'none']

    gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == 'mean':
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == 'sum':
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte

def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte

def get_rotation_translation_from_transform(transform):
    r"""Decompose transformation matrix into rotation matrix and translation vector.

    Args:
        transform (Tensor): (*, 4, 4)

    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation

def statistical_outlier_rm(pcd, num, std=1.0):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    cl, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=num, std_ratio=std)
    pcd_cl = np.array(cl.points)
    return pcd_cl

def debug_save_pcd(pcd, dir):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(dir, pcd_o3d)

def save_transformed_pcd(output_dict, data_dict, log_dir, level, norm_factor=1.0):
    transform = copy.deepcopy(data_dict['transform_raw'].squeeze(0))
    transform[:3, 3] = transform[:3, 3]
    if level == 'coarse':
        est_transform = copy.deepcopy(output_dict['coarse_trans'])
    elif level == 'refined':
        est_transform = copy.deepcopy(output_dict['refined_trans'])
    est_transform[:3, 3] = est_transform[:3, 3]

    src_points = data_dict['src_points'].squeeze(0)
    ref_points = data_dict['ref_points_raw'].squeeze(0)
    gt_src_points = apply_transform(src_points, transform).cpu().numpy()
    est_src_points = apply_transform(src_points, est_transform).cpu().numpy()

    src_pcd_plt = o3d.geometry.PointCloud()
    src_pcd_plt.points = o3d.utility.Vector3dVector(src_points.cpu().numpy())
    o3d.io.write_point_cloud(log_dir + "src_pcd_plt.ply", src_pcd_plt)
    ref_pcd_plt = o3d.geometry.PointCloud()
    ref_pcd_plt.points = o3d.utility.Vector3dVector(ref_points.cpu().numpy())
    o3d.io.write_point_cloud(log_dir + "ref_pcd_plt.ply", ref_pcd_plt)

    gt_tran_pcd_plt = o3d.geometry.PointCloud()
    gt_tran_pcd_plt.points = o3d.utility.Vector3dVector(gt_src_points)
    o3d.io.write_point_cloud(log_dir + "gt_tran_pcd_plt.ply", gt_tran_pcd_plt)
    est_tran_pcd_plt = o3d.geometry.PointCloud()
    est_tran_pcd_plt.points = o3d.utility.Vector3dVector(est_src_points)
    o3d.io.write_point_cloud(log_dir + level + "_est_tran_pcd_plt.ply", est_tran_pcd_plt)

    color_gt = np.array([[30, 255, 0] for i in range(gt_src_points.shape[0])])
    color_est = np.array([[0, 255, 255] for i in range(est_src_points.shape[0])])

    gt = np.concatenate((gt_src_points, color_gt), axis=1)
    est = np.concatenate((est_src_points, color_est), axis=1)
    cat = np.concatenate((gt, est), axis=0)

    return cat

def save_recon_pcd(output_dict, data_dict, log_dir):

    src_points = data_dict['src_points'].squeeze(0)
    tgt_points = data_dict['ref_points'].squeeze(0).cpu().numpy()
    transform = data_dict['transform_raw'].squeeze(0)
    gt_src_points = apply_transform(src_points, transform).cpu().numpy()
    recon = output_dict['recon'].squeeze(0).cpu().numpy()

    tgt_pcd_plt = o3d.geometry.PointCloud()
    tgt_pcd_plt.points = o3d.utility.Vector3dVector(tgt_points)
    o3d.io.write_point_cloud(log_dir + "tgt_pcd_plt.ply", tgt_pcd_plt)


    src_pcd_plt = o3d.geometry.PointCloud()
    src_pcd_plt.points = o3d.utility.Vector3dVector(gt_src_points)
    o3d.io.write_point_cloud(log_dir + "gt_src_pcd_plt.ply", src_pcd_plt)

    recon_pcd_plt = o3d.geometry.PointCloud()
    recon_pcd_plt.points = o3d.utility.Vector3dVector(recon)
    o3d.io.write_point_cloud(log_dir + "recon_pcd_plt.ply", recon_pcd_plt)

    color_src_gt = np.array([[0, 255, 0] for i in range(gt_src_points.shape[0])])
    color_src_recon = np.array([[255, 0, 0] for i in range(recon.shape[0])])
    color_tgt = np.array([[0, 0, 255] for i in range(tgt_points.shape[0])])

    tgt = np.concatenate((tgt_points, color_tgt), axis=1)
    src_gt = np.concatenate((gt_src_points, color_src_gt), axis=1)
    recon = np.concatenate((recon, color_src_recon), axis=1)
    vis = np.concatenate((src_gt, recon, tgt), axis=0)

    return vis

def save_traj(output_dict, data_dict, model_dir, traj_dir, norm_factor=1.0, residual_t=False):
    obj_id = data_dict['obj_id'].item()
    model_file = model_dir + "/obj_" + str(obj_id).zfill(6) + ".ply"
    obj_mesh = o3d.io.read_triangle_mesh(model_file)
    gt_transformation = data_dict['transform_raw'].squeeze(0)
    # scale the translation to 1000 times
    gt_rescale_transformation = copy.deepcopy(gt_transformation)
    gt_rescale_transformation[0:3, 3] = gt_transformation[0:3, 3] * 1000.0 / norm_factor
    gt_obj_mesh = copy.deepcopy(obj_mesh)
    gt_obj_mesh.transform(gt_rescale_transformation.cpu().numpy())
    # change the color of the mesh: green
    gt_obj_mesh.paint_uniform_color([0.0, 1.0, 0.0])
    o3d.io.write_triangle_mesh(traj_dir + "gt.ply", gt_obj_mesh)

    traj = output_dict['traj']
    for idx, transformation in enumerate(traj):
        hypotheses_mesh = o3d.geometry.TriangleMesh()
        for hypothesis_idx in range(transformation.shape[0]):

            step_transformation = transformation[hypothesis_idx, :, :].squeeze(0) # select the first hypothesis
            ortho6d = step_transformation[:6]
            if residual_t:
                trans = (step_transformation[6:] + output_dict['center_ref']).cpu() * 1000.0 / norm_factor
            else:
                trans = step_transformation[6:].cpu() * 1000.0 / norm_factor
            rot = compute_rotation_matrix_from_ortho6d(ortho6d.unsqueeze(0)).squeeze(0).cpu()
            transformation_matrix = torch.from_numpy(get_transform_from_rotation_translation(rot, trans).astype(np.float32))
            hypothesis_mesh = copy.deepcopy(obj_mesh)

            hypothesis_mesh.transform(transformation_matrix.numpy())
            hypotheses_mesh += hypothesis_mesh

        # make 2 meshes in one file
        vis_mesh = o3d.geometry.TriangleMesh()
        #vis_mesh += gt_obj_mesh
        vis_mesh += hypotheses_mesh
        # save the mesh
        vis_filename = traj_dir + str(idx) + ".ply"
        o3d.io.write_triangle_mesh(vis_filename, vis_mesh)




import csv

def write_result_csv(output_dict, data_dict, filepath_c, filepath_r, norm_factor=1.0):
    scene_id = data_dict['scene_id'].item()
    img_id = data_dict['img_id'].item()
    obj_id = data_dict['obj_id'].item()

    score = 1.0
    transform_c = output_dict['coarse_trans'].cpu().numpy()
    transform_r = output_dict['refined_trans'].cpu().numpy()
    rot_c = transform_c[:3, :3]
    rot_row_wise_c = rot_c.flatten()

    trans_c = transform_c[:3, 3] * 1000.0 / norm_factor

    rot_r = transform_r[:3, :3]
    rot_row_wise_r = rot_r.flatten()

    trans_r = transform_r[:3, 3] * 1000.0 / norm_factor
    
    time = 1.0

    with open(filepath_c, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([scene_id, img_id, obj_id, score, ' '.join(map(str, rot_row_wise_c)), ' '.join(map(str, trans_c)), time])

    with open(filepath_r, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([scene_id, img_id, obj_id, score, ' '.join(map(str, rot_row_wise_r)), ' '.join(map(str, trans_r)), time])
        

def update_category_loss(category_loss, result_dict):
    obj_id = result_dict['obj_id']
    category_loss[obj_id-1].append(result_dict['loss'])
    return category_loss

def print_mean_category_loss(category_loss):
    for obj_id in category_loss.keys():
        print("obj_id: ", obj_id+1, "mean loss: ", np.average(category_loss[obj_id]))

def save_category_loss(category_loss, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for obj_id in category_loss.keys():
            for loss in category_loss[obj_id]:
                writer.writerow([obj_id+1, loss])
                
def update_category_add(category_add_c, category_add_r, result_dict):
    obj_id = result_dict['obj_id']
    category_add_c[obj_id-1].append(result_dict['ADD_C'])
    category_add_r[obj_id-1].append(result_dict['ADD_R'])

    return category_add_c, category_add_r

def save_category_add(category_add, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for obj_id in category_add.keys():
            mean = np.average(category_add[obj_id])
            writer.writerow([obj_id+1, mean])