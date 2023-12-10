import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from posediff.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, "output", _C.exp_name)
_C.snapshot_encoder_dir = osp.join(_C.output_dir, "snapshots/encoder")
_C.snapshot_ddpm_dir = osp.join(_C.output_dir, "snapshots/ddpm")
_C.snapshot_recon_dir = osp.join(_C.output_dir, "snapshots/recon")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")
_C.result_pcd_dir = osp.join(_C.output_dir, "result/pcd")
_C.result_csv_dir = osp.join(_C.output_dir, "result/csv")

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_encoder_dir)
ensure_dir(_C.snapshot_ddpm_dir)
ensure_dir(_C.snapshot_recon_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.result_pcd_dir)
ensure_dir(_C.result_csv_dir)

# wandb ddpm
_C.wandb_ddpm = edict()
_C.wandb_ddpm.enable = True
_C.wandb_ddpm.project = "cordi_pose_base"
_C.wandb_ddpm.name = "lm8_pbr_b16_L_res_t_o6d_mh16_400step_norm_10_d512_add_dino_foldnet_8l_icp"

# wandb recon
_C.wandb_recon = edict()
_C.wandb_recon.enable = False
_C.wandb_recon.project = "cordi_recon_comp"
_C.wandb_recon.name = "lm12_pbr_b32_or100_foldnet_plane_k64_d512"

# data
_C.data = edict()
_C.data.dataset = "linemod"
_C.data.norm_factor = 10.0
_C.data.residual_t = True

# train data
_C.train = edict()
_C.train.batch_size = 16
_C.train.num_workers = 8

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8

# evaluation
_C.eval = edict()
_C.eval.rre_threshold = 1.0
_C.eval.rte_threshold = 0.1

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps = 1000
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.01
_C.optim.max_iteration = 500000
_C.optim.snapshot_steps = 5000
_C.optim.grad_acc_steps = 1

# model - DINO
_C.dino = edict()
_C.dino.arch = 'vit_base'
_C.dino.patch_size = 8
_C.dino.pretrained_weights = ''
_C.dino.checkpoint_key = "teacher"
_C.dino.output_dim = 768
_C.dino.vis = False

# model - Recon
_C.recon = edict()
_C.recon.encoder = 'foldnet'
_C.recon.k = 64
_C.recon.feat_dims = 512
_C.recon.shape = 'plane'
_C.recon.cls_emb = False


# model - DDPM
_C.ddpm = edict()
_C.ddpm.num_steps = 400
_C.ddpm.respacing = 'ddim100'
_C.ddpm.noise_schedule = 'linear'
_C.ddpm.rotation_type = 'ortho6d'
_C.ddpm.multi_hypothesis = 16
_C.ddpm.time_emb_dim = 256

# model - DDPM - Transformer
_C.ddpm_transformer = edict()
_C.ddpm_transformer.n_layers = 8
_C.ddpm_transformer.n_heads = 4
_C.ddpm_transformer.query_dimensions = 128
_C.ddpm_transformer.feed_forward_dimensions = 1024
_C.ddpm_transformer.fusion_type = 'add'



def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
