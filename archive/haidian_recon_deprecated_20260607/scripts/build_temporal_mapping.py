"""构建 temporal_mapping.json — 以 tianyi_sar 为锚点，匹配各源最近帧."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


def extract_date(filename: str) -> datetime | None:
    """从 20250101.tif 提取日期."""
    try:
        stem = Path(filename).stem
        return datetime.strptime(stem, "%Y%m%d")
    except ValueError:
        return None


def find_nearest_file(target_date: datetime, files: list[Path]) -> str | None:
    """找时间最近的文件，返回文件名（不含扩展名）."""
    best = None
    best_delta = timedelta(days=9999)
    for f in files:
        d = extract_date(f.name)
        if d is None:
            continue
        delta = abs(d - target_date)
        if delta < best_delta:
            best_delta = delta
            best = f.stem
    return best


def build_mapping(
    data_root: str = "data_raw/haidian/scenes",
    planet_root: str = "data_raw/beijing/planetscene",
    output_path: str = "haidian_recon/.cache/temporal_mapping.json",
    max_days: float | None = None,
) -> None:
    # 若未指定，读取配置中的 temporal_window_days
    if max_days is None:
        try:
            from haidian_recon.config import Config
            cfg = Config()
            max_days = getattr(cfg.data, "temporal_window_days", 5.5)
        except Exception:
            max_days = 5.5
    max_days = float(max_days)
    data_root = Path(data_root)
    planet_root = Path(planet_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_names = ["s2", "s1", "landsat", "planet"]
    anchor_source = "tianyi_sar"

    patches = sorted([p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("patch_")])

    mapping: dict = {}
    for patch_id in patches:
        anchor_dir = data_root / patch_id / anchor_source
        if not anchor_dir.exists():
            continue

        anchor_files = sorted(anchor_dir.glob("*.tif"))
        if not anchor_files:
            continue

        # 预加载其他源文件列表
        source_files = {}
        for src in source_names:
            if src == "planet":
                src_dir = planet_root / patch_id
            else:
                src_dir = data_root / patch_id / src
            if src_dir.exists():
                source_files[src] = sorted(src_dir.glob("*.tif"))
            else:
                source_files[src] = []

        entries = []
        for af in anchor_files:
            anchor_date = extract_date(af.name)
            if anchor_date is None:
                continue

            entry = {
                "anchor_date": af.stem,
                "sources": {},
            }
            for src in source_names:
                nearest = find_nearest_file(anchor_date, source_files[src])
                if nearest:
                    # 检查时间差
                    nd = datetime.strptime(nearest, "%Y%m%d")
                    delta = abs(nd - anchor_date).days
                    if delta <= max_days:
                        entry["sources"][src] = nearest
            entries.append(entry)

        if entries:
            mapping[patch_id] = entries

    output_path.write_text(json.dumps(mapping, indent=2))
    total_samples = sum(len(v) for v in mapping.values())
    print(f"Built mapping: {len(mapping)} patches, {total_samples} samples -> {output_path}")


if __name__ == "__main__":
    build_mapping()
