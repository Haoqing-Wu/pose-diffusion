import time
import torch
import torch.optim as optim

from IPython import embed

from config import make_cfg
from loss import DDPMEvaluator
from dataset import train_valid_data_loader
from posediff.engine.iter_based_trainer_ddpm import IterBasedDDPMTrainer
from posediff.utils.torch import build_warmup_cosine_lr_scheduler, load_pretrained_weights_dino
from posediff.modules.backbone.wrapper import create_posediff
from posediff.modules.backbone import vision_transformer as vits
from posediff.modules.recon.model import create_model
from posediff.modules.backbone.utils import visualize_attention


class DDPMTrainer(IterBasedDDPMTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_iteration=cfg.optim.max_iteration, snapshot_steps=cfg.optim.snapshot_steps)
        self.cfg = cfg
        # dataloader
        start_time = time.time()
        train_loader, val_loader, test_loader = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader, test_loader)

        # 3D encoder
        encoder_model = create_model(cfg).cuda() # encoder
        encoder_model = self.register_pretrained_model(encoder_model)
        # 2D encoder
        dino_model = vits.__dict__[cfg.dino.arch](patch_size=cfg.dino.patch_size, num_classes=0).cuda()
        dino_model = self.register_dino_model(dino_model)
        load_pretrained_weights_dino(
            self.dino_model, 
            cfg.dino.pretrained_weights, 
            cfg.dino.checkpoint_key, 
            cfg.dino.arch, 
            cfg.dino.patch_size
            )
        # Diffusion
        model = create_posediff(cfg).cuda() # ddpm
        model = self.register_model(model)

        # optimizer, scheduler
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = build_warmup_cosine_lr_scheduler(
            optimizer,
            total_steps=cfg.optim.max_iteration,
            warmup_steps=cfg.optim.warmup_steps,
            eta_init=cfg.optim.eta_init,
            eta_min=cfg.optim.eta_min,
            grad_acc_steps=cfg.optim.grad_acc_steps
            )
        self.register_scheduler(scheduler)
        # Evaluator
        self.evaluator = DDPMEvaluator(cfg).cuda()

    def train_step(self, iteration, data_dict):
        with torch.no_grad():
            feat_2d = self.dino_model(data_dict['rgb'])
            if self.cfg.dino.vis:
                attention = self.dino_model.get_last_selfattention(data_dict['rgb'])
                visualize_attention(attention, data_dict['rgb'])
            feat_3d = self.encoder_model.get_feat(data_dict).squeeze(1)
        data_dict['feat_2d'] = feat_2d
        data_dict['feat_3d'] = feat_3d
        result_dict = self.model.get_loss(data_dict)
        return result_dict

    def val_step(self, iteration, data_dict):
        feat_2d = self.dino_model(data_dict['rgb'])
        feat_3d = self.encoder_model.get_feat(data_dict).squeeze(1)
        data_dict['feat_2d'] = feat_2d.squeeze(0)
        data_dict['feat_3d'] = feat_3d.squeeze(0)
        output_dict = self.model.sample(data_dict)
        output_dict = self.model.refine(output_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        return output_dict, result_dict

def main():
    cfg = make_cfg()
    trainer = DDPMTrainer(cfg)
    trainer.run()



if __name__ == '__main__':
    main()