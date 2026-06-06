import re

log_path = '/workspace/xuannv/outputs/exp_dual_teacher_v3/train_20260604_160818.log'
pattern = r'\[(\d{2}:\d{2}:\d{2})\] Epoch (\d{3})/\d+ \| total=([\-\d.]+) recon=([\-\d.]+) cls=([\-\d.]+) var=([\-\d.]+) cov=([\-\d.]+) l2unif=([\-\d.]+) erank=([\-\d.]+) aef=\[sp=([\-\d.]+),gl=([\-\d.]+)\] olmo=\[sp=([\-\d.]+),gl=([\-\d.]+)\] lr=([\-\d.]+)'

evals = []
with open(log_path) as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            time_str, epoch, total, recon, cls, var, cov, l2unif, erank, aef_sp, aef_gl, olmo_sp, olmo_gl, lr = m.groups()
            evals.append({
                'epoch': int(epoch),
                'total': float(total),
                'recon': float(recon),
                'cls': float(cls),
                'var': float(var),
                'cov': float(cov),
                'l2unif': float(l2unif),
                'erank': float(erank),
                'aef_sp': float(aef_sp),
                'aef_gl': float(aef_gl),
                'olmo_sp': float(olmo_sp),
                'olmo_gl': float(olmo_gl),
                'lr': float(lr),
            })

# Also extract kNN eval results
knn_pattern = r'\[Eval\] epoch (\d+): kNN acc=([\d.]+) mIoU=([\d.]+)'
knn_results = {}
with open(log_path) as f:
    for line in f:
        m = re.search(knn_pattern, line)
        if m:
            epoch, acc, miou = m.groups()
            knn_results[int(epoch)] = {'acc': float(acc), 'miou': float(miou)}

print('=' * 90)
print('训练效果分析 - exp_dual_teacher_v3')
print('=' * 90)
print()
print('【Epoch 摘要】')
print(f"{'Epoch':>6} {'Total':>8} {'Recon':>7} {'Cls':>7} {'L2Unif':>8} {'Erank':>7} {'LR':>10} {'kNN-mIoU':>9}")
print('-' * 90)
for e in evals:
    miou = knn_results.get(e['epoch'], {}).get('miou', '-')
    miou_str = f'{miou:.4f}' if isinstance(miou, float) else '-'
    print(f"{e['epoch']:>6} {e['total']:>8.3f} {e['recon']:>7.3f} {e['cls']:>7.3f} {e['l2unif']:>8.3f} {e['erank']:>7.1f} {e['lr']:>10.6f} {miou_str:>9}")

print()
print('【关键趋势】')
print(f"- 总损失 (total): {evals[0]['total']:.3f} -> {evals[-1]['total']:.3f} (下降 {evals[0]['total'] - evals[-1]['total']:.3f})")
print(f"- 重建损失 (recon): {evals[0]['recon']:.3f} -> {evals[-1]['recon']:.3f}")
print(f"- 分类损失 (cls): {evals[0]['cls']:.3f} -> {evals[-1]['cls']:.3f}")
print(f"- L2 均匀性 (l2unif): {evals[0]['l2unif']:.3f} -> {evals[-1]['l2unif']:.3f} (更负=更好)")
print(f"- 有效秩 (erank): {evals[0]['erank']:.1f} -> {evals[-1]['erank']:.1f} (目标 ~32)")
print(f"- 学习率 (lr): {evals[0]['lr']:.6f} -> {evals[-1]['lr']:.6f}")
print()
print('【kNN mIoU 趋势】')
for epoch in sorted(knn_results.keys()):
    print(f"  Epoch {epoch:>2}: acc={knn_results[epoch]['acc']:.4f}, mIoU={knn_results[epoch]['miou']:.4f}")

best_miou_epoch = max(knn_results.keys(), key=lambda e: knn_results[e]['miou'])
best_miou = knn_results[best_miou_epoch]['miou']
print()
print(f"【最佳 kNN mIoU】Epoch {best_miou_epoch}: {best_miou:.4f}")
print(f"【AEF Baseline】 mIoU = 0.4053")
print(f"【差距】 {0.4053 - best_miou:.4f} ({(best_miou / 0.4053) * 100:.1f}% of baseline)")
print()

# Analyze distill trends
print('【蒸馏损失趋势】')
print(f"  AEF spatial:  {evals[0]['aef_sp']:.3f} -> {evals[-1]['aef_sp']:.3f}")
print(f"  AEF global:   {evals[0]['aef_gl']:.3f} -> {evals[-1]['aef_gl']:.3f}")
print(f"  Olmo spatial: {evals[0]['olmo_sp']:.3f} -> {evals[-1]['olmo_sp']:.3f}")
print(f"  Olmo global:  {evals[0]['olmo_gl']:.3f} -> {evals[-1]['olmo_gl']:.3f}")
