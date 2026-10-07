import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, lambda_):
        ctx.lambda_ = lambda_
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, input_tensor):
        return GradientReversalFunction.apply(input_tensor, self.lambda_)


class DomainAdversarialHead(nn.Module):
    def __init__(self, in_features, num_domains, hidden_dim=256, dropout=0.5, lambda_=1.0):
        super().__init__()
        layers = [GradientReversalLayer(lambda_)]
        if hidden_dim and hidden_dim > 0:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_domains))
        else:
            layers.append(nn.Linear(in_features, num_domains))
        self.classifier = nn.Sequential(*layers)

    def forward(self, features):
        return self.classifier(features)