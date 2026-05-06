from processor.part_attention_vit_processor import part_attention_vit_do_train_with_amp
from processor.ori_vit_processor_with_amp import ori_vit_do_train_with_amp
from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import loss as Patchloss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_single_training(cfg, class_mode, local_rank):
    split_log_name = cfg.LOG_NAME if class_mode in (None, 'all') else f"{cfg.LOG_NAME}_{class_mode}"
    train_loader = build_reid_train_loader(cfg, class_mode=class_mode)
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name, class_mode=class_mode)
    num_classes = len(train_loader.dataset.pids)
    model_name = cfg.MODEL.NAME
    model = make_model(cfg, modelname=model_name, num_class=num_classes, camera_num=None, view_num=None)
    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in cfg.MODEL.NAME:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad = False
        print("====== freeze patch_embed for stability ======")

    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)

    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion = Patchloss.Pedal(scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()
    if cfg.MODEL.SOFT_LABEL and cfg.MODEL.NAME == 'part_attention_vit':
        print("========using soft label========")

    output_dir = os.path.join(cfg.LOG_ROOT, split_log_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info("Training class split mode: {}".format(class_mode))

    do_train_dict = {
        'part_attention_vit': part_attention_vit_do_train_with_amp
    }
    if model_name not in do_train_dict.keys():
        ori_vit_do_train_with_amp(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query, local_rank,
            log_name=split_log_name,
        )
    else:
        do_train_dict[model_name](
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query, local_rank,
            patch_centers=patch_centers,
            pc_criterion=pc_criterion,
            log_name=split_log_name,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--class_split_mode",
        default="all",
        type=str,
        choices=["all", "traffic", "non_traffic", "dual"],
        help="Train a single split or both traffic/non-traffic models.",
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    requested_mode = args.class_split_mode or cfg.MODEL.CLASS_SPLIT_MODE
    if requested_mode == 'dual':
        for split_mode in ('traffic', 'non_traffic'):
            run_single_training(cfg, split_mode, args.local_rank)
    else:
        run_single_training(cfg, requested_mode, args.local_rank)
