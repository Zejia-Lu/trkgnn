#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 .lt (torch 保存的 DTrack 事件列表) 批量转换为 ROOT 文件（使用 uproot 写入）。
每个 .lt 文件被认为包含一个“事件列表”，而每个事件包含若干 DTrack，
每条 DTrack 持有 all_hits (N_hits x 3) 的命中坐标。
输出为一棵树：每个“事件”一条 entry，变量长度数组存放该事件所有命中，
并用 track_id 标明命中所属的轨迹索引。
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import torch

# 确保 DTrack 可被反序列化（unpickle）
from utility.DTrack import DTrack  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert torch .lt (list of DTrack per event) to a ROOT tree with jagged branches."
    )
    p.add_argument(
        "-i", "--input-dir",
        default="/lustre/collider/wanghuayang/DeepLearning/Tracking/trkgnn/run/HPS_withBG/50/apply.momentum.rec.0p999/DigitizedRecTrk",
        help="包含 .lt 文件的目录（默认与原脚本相同）",
    )
    p.add_argument(
        "-p", "--file-pattern",
        default="tracks_*.lt",
        help="文件通配符（如 'tracks_*.lt'）",
    )
    p.add_argument(
        "-o", "--output-root",
        default="tracks_all.root",
        help="输出 ROOT 文件路径",
    )
    p.add_argument(
        "-t", "--tree-name",
        default="events",
        help="输出 TTree 名称（默认：events）",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索 input-dir（使用 ** 通配）",
    )
    p.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="最多处理的文件数（用于快速测试）",
    )
    p.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="最多写入的事件条目数（用于快速测试）",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在则覆盖",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安静模式：减少日志输出",
    )
    return p.parse_args()


def discover_files(input_dir: str, pattern: str, recursive: bool) -> list[str]:
    if recursive and "**" not in pattern:
        # 自动升级为递归模式
        pattern = f"**/{pattern}"
    search_glob = str(Path(input_dir) / pattern)
    files = sorted(glob.glob(search_glob, recursive=recursive))
    return files


def main() -> int:
    args = parse_args()

    if not args.quiet:
        print("=== Configuration ===")
        print(f"input_dir     : {args.input_dir}")
        print(f"file_pattern  : {args.file_pattern}")
        print(f"output_root   : {args.output_root}")
        print(f"tree_name     : {args.tree_name}")
        print(f"recursive     : {args.recursive}")
        print(f"limit_files   : {args.limit_files}")
        print(f"limit_events  : {args.limit_events}")
        print(f"overwrite     : {args.overwrite}")

    # 输出文件存在时的处理
    out_path = Path(args.output_root)
    if out_path.exists():
        if args.overwrite:
            if not args.quiet:
                print(f"[Info] Overwrite enabled, removing existing file: {out_path}")
            out_path.unlink()
        else:
            print(f"[Error] Output file already exists: {out_path} (use --overwrite to replace)", file=sys.stderr)
            return 2

    # 寻找输入文件
    torch_files = discover_files(args.input_dir, args.file_pattern, args.recursive)
    if args.limit_files is not None:
        torch_files = torch_files[: args.limit_files]

    if not args.quiet:
        print(f"[Info] Found {len(torch_files)} file(s).")

    if len(torch_files) == 0:
        print("[Warning] No input files matched. Nothing to do.")
        # 创建空 ROOT 也没有意义，直接返回
        return 0

    # 创建 ROOT 文件并逐步写入
    total_events_written = 0
    total_events_seen = 0

    with uproot.recreate(args.output_root) as fout:
        # 我们第一次写入时自动创建树；之后使用 .extend 追加
        tree_created = False

        for idx, fpath in enumerate(torch_files):
            if not args.quiet:
                print(f"[{idx+1}/{len(torch_files)}] Loading: {fpath}")

            # 每个 .lt：list[ list[DTrack] ] ——> 每个元素为一个“事件”的轨迹列表
            events = torch.load(fpath, map_location="cpu")

            for ievt, event_tracks in enumerate(events):
                total_events_seen += 1

                # 可选：限制总事件数
                if args.limit_events is not None and total_events_written >= args.limit_events:
                    if not args.quiet:
                        print(f"[Info] Reached limit_events={args.limit_events}, stop.")
                    if not tree_created:
                        # 没有任何可写入的事件，确保至少创建空文件
                        pass
                    # 正常结束
                    print(f"✅ Saved {total_events_written} events to {args.output_root}")
                    return 0

                if not event_tracks:
                    continue  # 跳过空事件

                # 同一事件中的所有轨迹
                try:
                    event_num = getattr(event_tracks[0], "evt_num", total_events_seen)
                except Exception:
                    event_num = total_events_seen  # 兜底

                n_tracks = len(event_tracks)

                hit_x, hit_y, hit_z, track_id = [], [], [], []

                for t_idx, trk in enumerate(event_tracks):
                    hits = getattr(trk, "all_hits", None)
                    if hits is None:
                        continue
                    hits = np.asarray(hits)
                    if hits.size == 0:
                        continue
                    # 期望形状 (N_hits, 3)
                    # 逐条追加至变量长度数组
                    for h in hits:
                        # 防御性转换
                        hit_x.append(float(h[0]))
                        hit_y.append(float(h[1]))
                        hit_z.append(float(h[2]))
                        track_id.append(int(t_idx))

                # 若该事件无有效命中则跳过
                if len(hit_x) == 0:
                    continue

                # 组装 Awkward 数组（每个 entry 包一个 list）
                # 结构：标量列 event_num, n_tracks；记录列 hit.{x,y,z,track_id} 为变长数组
                payload = {
                    "event_num": ak.Array([int(event_num)]),
                    "n_tracks": ak.Array([int(n_tracks)]),
                    "hit": ak.zip(
                        {
                            "x": ak.Array([hit_x]),
                            "y": ak.Array([hit_y]),
                            "z": ak.Array([hit_z]),
                            "track_id": ak.Array([track_id]),
                        }
                    ),
                }

                if not tree_created:
                    fout[args.tree_name] = payload
                    tree_created = True
                else:
                    fout[args.tree_name].extend(payload)

                total_events_written += 1

            # 释放当前批的内存（尤其是巨大的 events）
            del events

    print(f"✅ Saved {total_events_written} events to {args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
