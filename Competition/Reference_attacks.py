"""
Reference Attack Methods for COMP219 Assignment
Author: Lingfang Li
"""

import torch
import torch.nn.functional as F

EPSILON = 0.11  # assignment constraint

################################################################################################
## Fast attacks (recommended for debugging)
################################################################################################

def fgsm_attack(model, X, y, device):
    """Fast Gradient Sign Method - one step attack"""
    X_adv = X.clone().detach().requires_grad_(True)
    output = model(X_adv)
    loss = F.nll_loss(output, y)
    loss.backward()

    X_adv = X + EPSILON * X_adv.grad.sign()
    X_adv = torch.clamp(X_adv, 0.0, 1.0)
    return X_adv.detach()


def pgd5_attack(model, X, y, device):
    """Projected Gradient Descent - 5 steps (medium strength, ~2s/batch)"""
    alpha = 0.03
    X_adv = X.clone().detach() + torch.empty_like(X).uniform_(-EPSILON, EPSILON)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)

    for _ in range(5):
        X_adv.requires_grad_(True)
        output = model(X_adv)
        loss = F.nll_loss(output, y)
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            X_adv = torch.max(torch.min(X_adv, X + EPSILON), X - EPSILON)
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.detach()


def pgd20_attack(model, X, y, device):
    """Projected Gradient Descent - 20 steps (strong, ~8s/batch)"""
    alpha = 0.01
    X_adv = X.clone().detach() + torch.empty_like(X).uniform_(-EPSILON, EPSILON)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)

    for _ in range(20):
        X_adv.requires_grad_(True)
        output = model(X_adv)
        loss = F.nll_loss(output, y)
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            X_adv = torch.max(torch.min(X_adv, X + EPSILON), X - EPSILON)
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.detach()

