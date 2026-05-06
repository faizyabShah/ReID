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

    labels = labels.view(-1)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = ~is_pos

    eye = torch.eye(N, dtype=torch.bool, device=dist_mat.device)
    is_pos = is_pos & ~eye

    # Keep the matrix shape intact so rows with different numbers of positives
    # cannot break the hard-mining step.
    pos_dist = dist_mat.masked_fill(~is_pos, float('-inf'))
    dist_ap, relative_p_inds = torch.max(pos_dist, dim=1, keepdim=True)
    no_pos = ~torch.isfinite(dist_ap)
    if no_pos.any():
      dist_ap = dist_ap.clone()
      relative_p_inds = relative_p_inds.clone()
      dist_ap[no_pos] = 0.0
      relative_p_inds[no_pos] = torch.arange(N, device=dist_mat.device, dtype=relative_p_inds.dtype).view(-1, 1)[no_pos]

    # If class_labels provided, split negatives into same-class and other-class negatives
    if class_labels is None:
      # `dist_an` means distance(anchor, negative)
      # both `dist_an` and `relative_n_inds` with shape [N, 1]
      neg_dist = dist_mat.masked_fill(~is_neg, float('inf'))
      dist_an, relative_n_inds = torch.min(neg_dist, dim=1, keepdim=True)
      no_neg = ~torch.isfinite(dist_an)
      if no_neg.any():
        dist_an = dist_an.clone()
        relative_n_inds = relative_n_inds.clone()
        dist_an[no_neg] = 0.0
        relative_n_inds[no_neg] = torch.arange(N, device=dist_mat.device, dtype=relative_n_inds.dtype).view(-1, 1)[no_neg]
      dist_ap = dist_ap.squeeze(1)
      dist_an = dist_an.squeeze(1)
    else:
      # build mask for same semantic class negatives
      cls = class_labels.view(-1)
      is_same_class = cls.expand(N, N).eq(cls.expand(N, N).t())
      # negatives that are same semantic class but different pid
      is_neg_sameclass = is_neg & is_same_class
      # negatives from other semantic classes
      is_neg_otherclass = is_neg & (~is_same_class)

      same_dist = dist_mat.masked_fill(~is_neg_sameclass, float('inf'))
      other_dist = dist_mat.masked_fill(~is_neg_otherclass, float('inf'))
      dist_an_same, relative_n_inds_same = torch.min(same_dist, dim=1, keepdim=True)
      dist_an_other, relative_n_inds_other = torch.min(other_dist, dim=1, keepdim=True)

      same_valid = torch.isfinite(dist_an_same)
      other_valid = torch.isfinite(dist_an_other)
      dist_an = torch.zeros_like(dist_an_same)
      relative_n_inds = torch.zeros_like(relative_n_inds_same)

      both_valid = same_valid & other_valid
      if both_valid.any():
        dist_an[both_valid] = 0.5 * dist_an_same[both_valid] + 0.5 * dist_an_other[both_valid]
        relative_n_inds[both_valid] = relative_n_inds_other[both_valid]

      only_same = same_valid & ~other_valid
      if only_same.any():
        dist_an[only_same] = dist_an_same[only_same]
        relative_n_inds[only_same] = relative_n_inds_same[only_same]

      only_other = other_valid & ~same_valid
      if only_other.any():
        dist_an[only_other] = dist_an_other[only_other]
        relative_n_inds[only_other] = relative_n_inds_other[only_other]

      neither = ~same_valid & ~other_valid
      if neither.any():
        neg_dist = dist_mat.masked_fill(~is_neg, float('inf'))
        fallback_dist, fallback_inds = torch.min(neg_dist, dim=1, keepdim=True)
        dist_an[neither] = fallback_dist[neither]
        relative_n_inds[neither] = fallback_inds[neither]

      dist_ap = dist_ap.squeeze(1)
      dist_an = dist_an.squeeze(1)

    if return_inds:
        p_inds = relative_p_inds.squeeze(1)
        n_inds = relative_n_inds.squeeze(1)
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


