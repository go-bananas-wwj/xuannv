"""
分析为什么 uniformity loss 无法阻止 dimensional collapse
"""
import torch
import math

# 模拟 64D embedding，但只使用 4 个维度（dimensional collapse）
N, D = 32, 64
# 4 个活跃维度，60 个死亡维度
active_dims = 4
x_collapsed = torch.randn(N, active_dims)  # 4D 子空间内随机
x_collapsed = torch.cat([x_collapsed, torch.zeros(N, D - active_dims)], dim=1)

# 模拟良好分散的 64D embedding
x_uniform = torch.randn(N, D)

def raw_uniformity_loss(embeddings):
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    z = embeddings - embeddings.mean(dim=0)
    global_std = z.std() + 1e-4
    z = z / global_std
    N = z.shape[0]
    D = z.shape[1]
    t = 2.0 / D
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    pair_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    sq_pdist_pairs = sq_pdist[pair_mask]
    loss = torch.logsumexp(-t * sq_pdist_pairs, dim=0) - math.log(sq_pdist_pairs.shape[0])
    return loss

def erank_maximization_loss(x):
    N, D = x.shape
    if N < 2:
        return x.new_tensor(0.0)
    x = x - x.mean(0, keepdim=True)
    col_var = x.pow(2).mean(dim=0).clamp(min=1e-8)
    probs = col_var / col_var.sum()
    entropy = -(probs * probs.log()).sum()
    max_entropy = math.log(float(D))
    return (max_entropy - entropy).clamp(min=0.0)

def variance_regularizer(x, min_std=1.0, eps=1e-4):
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(torch.relu(min_std - std))

def covariance_loss(x):
    N = x.shape[0]
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / D

def decorrelation_loss(x):
    N = x.shape[0]
    eps = 1e-4
    x = x - x.mean(0, keepdim=True)
    std = torch.sqrt(x.var(dim=0) + eps)
    x_norm = x / (std + 1e-4)
    c = (x_norm.T @ x_norm) / N
    identity = torch.eye(D)
    return (c - identity).pow(2).sum()

# 测试各种损失在 collapsed vs uniform 上的表现
print("=" * 60)
print(f"N={N}, D={D}, active_dims={active_dims}")
print(f"Collapsed embedding: first {active_dims} dims random, rest 0")
print(f"Uniform embedding: all {D} dims random")
print("=" * 60)

for name, x in [("Collapsed", x_collapsed), ("Uniform", x_uniform)]:
    print(f"\n{name}:")
    
    # raw_uniformity
    x_req = x.clone().requires_grad_(True)
    loss = raw_uniformity_loss(x_req)
    loss.backward()
    grad_norm = x_req.grad.norm()
    print(f"  raw_uniformity: {loss.item():.4f}  grad_norm: {grad_norm:.4f}")
    
    # erank
    x_req = x.clone().requires_grad_(True)
    loss = erank_maximization_loss(x_req)
    loss.backward()
    grad_norm = x_req.grad.norm()
    print(f"  erank_loss:     {loss.item():.4f}  grad_norm: {grad_norm:.4f}")
    
    # variance
    x_req = x.clone().requires_grad_(True)
    loss = variance_regularizer(x_req)
    loss.backward()
    grad_norm = x_req.grad.norm()
    print(f"  var_loss:       {loss.item():.4f}  grad_norm: {grad_norm:.4f}")
    
    # covariance
    x_req = x.clone().requires_grad_(True)
    loss = covariance_loss(x_req)
    loss.backward()
    grad_norm = x_req.grad.norm()
    print(f"  cov_loss:       {loss.item():.4f}  grad_norm: {grad_norm:.4f}")
    
    # decorrelation
    x_req = x.clone().requires_grad_(True)
    loss = decorrelation_loss(x_req)
    loss.backward()
    grad_norm = x_req.grad.norm()
    print(f"  decorr_loss:    {loss.item():.4f}  grad_norm: {grad_norm:.4f}")
    
    # 计算 erank
    x_centered = x - x.mean(0, keepdim=True)
    cov = (x_centered.T @ x_centered) / (N - 1)
    eigs = torch.linalg.eigvalsh(cov)
    eigs = eigs / (eigs.sum() + 1e-8)
    erank = torch.exp(-(eigs * torch.log(eigs + 1e-8)).sum()).item()
    print(f"  erank:          {erank:.2f}")
