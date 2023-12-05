import torch
import torch.nn as nn

from posediff.datasets.registration.linemod.bop_utils import apply_transform
from scipy.spatial.transform import Rotation
from posediff.datasets.registration.linemod.bop_utils import *
from sklearn.neighbors import KDTree



class ChamferLoss(nn.Module):
    def __init__(self, cfg):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.norm_factor = cfg.data.norm_factor

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def loss(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2
    
    def forward(self, output_dict, data_dict):
        obj_id = data_dict['obj_id']
        recon = output_dict['recon']
        src_points = data_dict['src_points']
        transform = data_dict['transform']
        gt_src_points = apply_transform(src_points, transform)
        loss = self.loss(recon, gt_src_points)

        return {
            'loss': loss,
            #'obj_id': obj_id
        }


    
class DDPMEvaluator(nn.Module):
    def __init__(self, cfg):
        super(DDPMEvaluator, self).__init__()
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold

  
    @torch.no_grad()
    def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
        R_gt, t_gt = pose_gt
        R_pred, t_pred = pose_pred
        count = R_gt.shape[0]
        mean_distances = np.zeros((count,), dtype=np.float32)
        for i in range(count):
            pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]        
            pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
            distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
            mean_distances[i] = np.mean(distance)            

        threshold = diameter * percentage
        score = (mean_distances < threshold).sum() / count
        return score

    @torch.no_grad()
    def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
        R_gt, t_gt = pose_gt
        R_pred, t_pred = pose_pred

        count = R_gt.shape[0]
        mean_distances = np.zeros((count,), dtype=np.float32)
        for i in range(count):
            if np.isnan(np.sum(t_pred[i])):
                mean_distances[i] = np.inf
                continue
            pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
            pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
            kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
            distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
            mean_distances[i] = np.mean(distance)
        threshold = diameter * percentage
        score = (mean_distances < threshold).sum() / count
        return score

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict, level):
        obj_id = data_dict['obj_id'].cpu().numpy()[0]
        info_file = "./data/lm/models/models_info.json"
        with open(info_file, 'r') as file:
            model_info = json.load(file)[str(obj_id)]
        diameter = model_info['diameter'] / 1000.0
        src_points = output_dict['src_points']
        
        transform = data_dict['transform_raw'].squeeze(0)
        if level == 'coarse':
            est_transform = output_dict['coarse_trans']
        elif level == 'refined':
            est_transform = output_dict['refined_trans']

        
        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.acceptance_rre), torch.lt(rte, self.acceptance_rte)).float()

        gt_src_points = apply_transform(src_points, transform)
        est_src_points = apply_transform(src_points, est_transform)
        rmse = torch.linalg.norm(est_src_points - gt_src_points, dim=1).mean()
        
        if obj_id in [3, 10, 11]:
            kdt = KDTree(gt_src_points.cpu().numpy(), metric='euclidean')
            distance, _ = kdt.query(est_src_points.cpu().numpy(), k=1)
            mean_distance = np.mean(distance)
            add_score = mean_distance < diameter * 0.1 
        else:
            add_score = rmse < diameter * 0.1
        return rre, rte, rmse, recall, add_score

    def forward(self, output_dict, data_dict):
        rre_c, rte_c, rmse_c, recall_c, add_c = self.evaluate_registration(output_dict, data_dict, 'coarse')
        rre_r, rte_r, rmse_r, recall_r, add_r = self.evaluate_registration(output_dict, data_dict, 'refined')
        return {
            'RRE_C': rre_c,
            'RTE_C': rte_c,
            'RMSE_C': rmse_c,
            'RR_C': recall_c,
            'ADD_C': add_c,
            'RRE_R': rre_r,
            'RTE_R': rte_r,
            'RMSE_R': rmse_r,
            'RR_R': recall_r,
            'ADD_R': add_r,
            'Var': output_dict['var_rt'],
            #'obj_id': data_dict['obj_id'].cpu().numpy()[0]
        }