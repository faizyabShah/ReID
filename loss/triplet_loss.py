from cProfile import label
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, class_labels=None, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
      dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)

    # If class_labels provided, split negatives into same-class and other-class negatives
    if class_labels is None:
      # `dist_an` means distance(anchor, negative)
      # both `dist_an` and `relative_n_inds` with shape [N, 1]
      dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
      dist_ap = dist_ap.squeeze(1)
      dist_an = dist_an.squeeze(1)
    else:
      # build mask for same semantic class negatives
      cls = class_labels
      is_same_class = cls.expand(N, N).eq(cls.expand(N, N).t())
      # negatives that are same semantic class but different pid
      is_neg_sameclass = is_neg & is_same_class
      # negatives from other semantic classes
      is_neg_otherclass = is_neg & (~is_same_class)

      dist_an_same = None
      dist_an_other = None
      relative_n_inds_same = None
      relative_n_inds_other = None

      # handle same-class negatives
      if is_neg_sameclass.any():
        tmp_same = dist_mat.clone()
        mask_same = ~is_neg_sameclass
        tmp_same[mask_same] = float('inf')
        dist_an_same_vals, relative_n_inds_same = torch.min(tmp_same, dim=1, keepdim=True)
        dist_an_same = dist_an_same_vals.squeeze(1)

      # handle other-class negatives
      if is_neg_otherclass.any():
        tmp_other = dist_mat.clone()
        mask_other = ~is_neg_otherclass
        tmp_other[mask_other] = float('inf')
        dist_an_other_vals, relative_n_inds_other = torch.min(tmp_other, dim=1, keepdim=True)
        dist_an_other = dist_an_other_vals.squeeze(1)

      # Combine: prefer averaging same-class and other-class hard negatives when both available
      if (dist_an_same is not None) and (dist_an_other is not None):
        dist_an = 0.5 * dist_an_same + 0.5 * dist_an_other
      elif dist_an_same is not None:
        dist_an = dist_an_same
      elif dist_an_other is not None:
        dist_an = dist_an_other
      else:
        # fallback to global negatives
        dist_an, relative_n_inds = torch.min(
          dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        dist_an = dist_an.squeeze(1)

      dist_ap = dist_ap.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
          ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        if class_labels is None:
          n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        else:
          # choose other-class negative index when available else same-class
          if relative_n_inds_other is not None:
            n_inds = torch.gather(
              ind[is_neg_otherclass].contiguous().view(N, -1), 1, relative_n_inds_other.data)
          elif relative_n_inds_same is not None:
            n_inds = torch.gather(
              ind[is_neg_sameclass].contiguous().view(N, -1), 1, relative_n_inds_same.data)
          else:
            n_inds = torch.gather(
              ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, class_labels=None, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
      dist_ap, dist_an = hard_example_mining(dist_mat, labels, class_labels=class_labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            # min_mat = dist_an.new().resize_as_(dist_an).fill_(-85)
            # input = max(min_mat, dist_an - dist_ap)
            input = dist_an - dist_ap
            loss = self.ranking_loss(input, y)
        return loss, dist_ap, dist_an


