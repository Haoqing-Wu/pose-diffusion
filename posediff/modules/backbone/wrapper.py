import torch
import open3d as o3d

from torch.nn import Module, Linear, ReLU
from posediff.modules.backbone.ddpm import *
from posediff.modules.backbone.transformer import *
from posediff.datasets.registration.linemod.bop_utils import *
from posediff.modules.diffusion import create_diffusion
from posediff.modules.backbone.rotation_tools import compute_rotation_matrix_from_ortho6d

class DiffusionWrapper(Module):

    def __init__(self, cfg):
        super(DiffusionWrapper, self).__init__()
        self.cfg = cfg
        self.multi_hypothesis = cfg.ddpm.multi_hypothesis
        self.rotation_type = cfg.ddpm.rotation_type
        self.norm_factor = cfg.data.norm_factor
        self.residual_t = cfg.data.residual_t

        self.diffusion_new = create_diffusion(
            timestep_respacing=cfg.ddpm.respacing,
            noise_schedule=cfg.ddpm.noise_schedule, 
            diffusion_steps=cfg.ddpm.num_steps
            )
        
        self.net = transformer(
                n_layers=cfg.ddpm_transformer.n_layers,
                n_heads=cfg.ddpm_transformer.n_heads,
                query_dimensions=cfg.ddpm_transformer.query_dimensions,
                time_emb_dim=cfg.ddpm.time_emb_dim,
                dino_emb_dim=cfg.dino.output_dim,
                recon_emb_dim=cfg.recon.feat_dims,
                fusion_type=cfg.ddpm_transformer.fusion_type,
            )


    def get_loss(self, d_dict):

        feat_2d = d_dict.get('feat_2d')
        feat_3d = d_dict.get('feat_3d')
        rt = d_dict.get('rt').unsqueeze(1)
        feats = {}
        feats['feat_2d'] = feat_2d
        feats['feat_3d'] = feat_3d
        t = torch.randint(0, self.diffusion_new.num_timesteps, (rt.shape[0],), device='cuda')
        loss_dict = self.diffusion_new.training_losses(self.net, rt, t, feats)
        loss = loss_dict["loss"].mean()
        return {'loss': loss}
    
    def sample(self, d_dict):

        feat_2d = d_dict.get('feat_2d')
        feat_3d = d_dict.get('feat_3d')
        feats = {}
        feats['feat_2d'] = feat_2d.repeat(self.multi_hypothesis, 1)
        feats['feat_3d'] = feat_3d.repeat(self.multi_hypothesis, 1)

        rt_T = torch.randn_like(d_dict.get('rt').repeat(self.multi_hypothesis, 1)).cuda().unsqueeze(1)

        traj = self.diffusion_new.p_sample_loop(
            self.net, rt_T.shape, rt_T, clip_denoised=False, model_kwargs=feats, progress=True, device='cuda'
        )
        pred_rt = traj[-1].cpu()
        pred_rt = pred_rt.squeeze(1)
        mean_rt = pred_rt.mean(dim=0)
        var_rt = pred_rt.var(dim=0)
        mean_var_rt = var_rt.mean(dim=0)

        return {
            'ref_points': d_dict.get('ref_points').squeeze(0),
            'ref_points_raw': d_dict.get('ref_points_raw').squeeze(0),
            'src_points': d_dict.get('src_points').squeeze(0),
            'center_ref': d_dict.get('center_ref').squeeze(0),
            'pred_rt': mean_rt,
            'var_rt': mean_var_rt,
            'traj': traj,
            }

    def refine(self, output_dict):

        ref_points = output_dict.get('ref_points_raw')
        src_points = output_dict.get('src_points')
        rt_init = output_dict.get('pred_rt')
        center_ref = output_dict.get('center_ref').cpu()
      
        if self.rotation_type == 'quat':
            quat = rt_init[:4]
            trans = rt_init[4:]
            if self.residual_t:
                trans+=center_ref
            r = Rotation.from_quat(quat)
            rot = r.as_matrix()
        elif self.rotation_type == 'mrp':
            mrp = rt_init[:3]
            trans = rt_init[3:]
            if self.residual_t:
                trans+=center_ref
            r = Rotation.from_mrp(mrp)
            rot = r.as_matrix()
        elif self.rotation_type == 'ortho6d':
            ortho6d = rt_init[:6]
            trans = rt_init[6:]
            if self.residual_t:
                trans+=center_ref
            # TODO: rewrite this part
            rot = compute_rotation_matrix_from_ortho6d(ortho6d.unsqueeze(0).cuda()).squeeze(0).cpu()
        
        init_trans = torch.from_numpy(get_transform_from_rotation_translation(rot, trans).astype(np.float32)).cuda()
        init_trans_icp = torch.from_numpy(get_transform_from_rotation_translation(rot, trans/self.norm_factor).astype(np.float32)).cuda()
        ref_points = ref_points.cpu().numpy()

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_points.cpu().numpy())
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(ref_points)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, 0.005, init_trans_icp.cpu().numpy(),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 20000))
        refined_trans = torch.from_numpy(reg_p2p.transformation.astype(np.float32)).cuda()
        refined_trans[:3, 3] = refined_trans[:3, 3] * self.norm_factor
        output_dict['refined_trans'] = refined_trans
        output_dict['coarse_trans'] = init_trans   
        return output_dict

def create_posediff(cfg):
    model = DiffusionWrapper(cfg)
    return model
