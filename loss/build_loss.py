import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS

feat_dim_dict = {
    'local_attention_vit': 768,
    'vit': 768,
    'resnet18': 512,
    'resnet34': 512
}

def build_loss(cfg, num_classes):
    name = cfg.MODEL.NAME
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.NAME not in feat_dim_dict.keys():
        feat_dim = 2048
    else:
        feat_dim = feat_dim_dict[cfg.MODEL.NAME]
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if name == 'local_attention_vit' and cfg.MODEL.PC_LOSS:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            xent = CE_LS(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax': # softmax loss only
        def loss_func(score, feat, target, class_labels=None):
            return F.cross_entropy(score, target)

    # softmax & triplet
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet' or 'GS':
        def loss_func(score, feat, target, target_cam=None, domains=None, t_domains=None, all_posvid=None, soft_label=False, soft_weight=0.1, soft_lambda=0.2, class_labels=None, domain_logits=None, return_components=False, **kwargs):
            domain_loss = 0.0
            if getattr(cfg.MODEL, 'DOMAIN_ADV', None) is not None and cfg.MODEL.DOMAIN_ADV.ENABLED and domain_logits is not None and target_cam is not None:
                if isinstance(domain_logits, list):
                    domain_loss = sum(F.cross_entropy(logit, target_cam) for logit in domain_logits) / len(domain_logits)
                else:
                    domain_loss = F.cross_entropy(domain_logits, target_cam)

            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if name == 'local_attention_vit' and cfg.MODEL.PC_LOSS:
                        ID_LOSS = xent(score, target, all_posvid=all_posvid, soft_label=soft_label,soft_weight=soft_weight, soft_lambda=soft_lambda)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                TRI_LOSS = triplet(feat, target, class_labels=class_labels)[0]
                # DOMAIN_LOSS = xent(domains, t_domains)
                total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                               cfg.MODEL.DOMAIN_ADV.LAMBDA * domain_loss
                if return_components:
                    return total_loss, {
                        'id_loss': ID_LOSS.detach(),
                        'tri_loss': TRI_LOSS.detach(),
                        'domain_loss': domain_loss.detach() if torch.is_tensor(domain_loss) else domain_loss,
                    }
                return total_loss
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    id_loss = xent(score, target)
                    tri_loss = triplet(feat, target, class_labels=class_labels)[0]
                    center_loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                    total_loss = id_loss + tri_loss + center_loss + cfg.MODEL.DOMAIN_ADV.LAMBDA * domain_loss
                    if return_components:
                        return total_loss, {
                            'id_loss': id_loss.detach(),
                            'tri_loss': tri_loss.detach(),
                            'center_loss': center_loss.detach(),
                            'domain_loss': domain_loss.detach() if torch.is_tensor(domain_loss) else domain_loss,
                        }
                    return total_loss
                else:
                    id_loss = F.cross_entropy(score, target)
                    tri_loss = triplet(feat, target, class_labels=class_labels)[0]
                    center_loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                    total_loss = id_loss + tri_loss + center_loss + cfg.MODEL.DOMAIN_ADV.LAMBDA * domain_loss
                    if return_components:
                        return total_loss, {
                            'id_loss': id_loss.detach(),
                            'tri_loss': tri_loss.detach(),
                            'center_loss': center_loss.detach(),
                            'domain_loss': domain_loss.detach() if torch.is_tensor(domain_loss) else domain_loss,
                        }
                    return total_loss
            else:
                print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                    'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


