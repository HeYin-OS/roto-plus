"""首次部署初始化脚本：拉取 sam2_git 源码（固定 commit）+ 下载 SAM2 checkpoint。

uv 管不到的两样东西在这里补齐，保证可复现：
  1. sam2_git/  —— SAM2 官方仓库，钉在验证过的 commit 2b90b9f
  2. sam2_git/checkpoints/*.pt —— base_plus 与 large 两个 2.1 checkpoint

幂等：已存在的文件/已检出的 commit 会跳过，可重复运行。

用法：
  uv run python scripts/init_sam2.py            # 完整初始化
  uv run python scripts/init_sam2.py --large-only  # 只下 large checkpoint
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM2_DIR = PROJECT_ROOT / "sam2_git"

# 已在 RTX 4070 + torch 2.5.1+cu121 上验证过的官方仓库 commit（2024.10 版）
SAM2_REPO = "https://github.com/facebookresearch/sam2.git"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

# 与官方 checkpoints/download_ckpts.sh 保持一致（SAM2.1, 092824 发布）
CKPT_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
CHECKPOINTS = {
    "sam2.1_hiera_base_plus.pt": "base_plus（mask_predictor 默认）",
    "sam2.1_hiera_large.pt": "large（备选/测试用）",
}


def sh(*cmd: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """运行子进程，失败即退出并打印输出。"""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"❌ 命令失败: {' '.join(cmd)}")
    return result


def clone_or_checkout() -> None:
    """clone SAM2 仓库并钉到指定 commit（幂等）。"""
    if (SAM2_DIR / "sam2" / "build_sam.py").exists():
        print(f"✅ sam2_git 已存在，跳过 clone: {SAM2_DIR}")
    else:
        print(f"⏳ clone {SAM2_REPO} -> {SAM2_DIR}")
        sh("git", "clone", SAM2_REPO, str(SAM2_DIR))

    current = sh("git", "rev-parse", "HEAD", cwd=SAM2_DIR).stdout.strip()
    if current == SAM2_COMMIT:
        print(f"✅ 已在目标 commit: {SAM2_COMMIT[:8]}")
        return
    print(f"⏳ checkout {SAM2_COMMIT[:8]}（当前 {current[:8]}）")
    sh("git", "checkout", SAM2_COMMIT, cwd=SAM2_DIR)


def download_checkpoints(only_large: bool = False) -> None:
    """下载 SAM2.1 checkpoint（幂等，跳过已存在文件）。"""
    ckpt_dir = SAM2_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, desc in CHECKPOINTS.items():
        if only_large and "large" not in name:
            continue
        dest = ckpt_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✅ 已存在，跳过: {dest.name}")
            continue
        url = f"{CKPT_BASE_URL}/{name}"
        print(f"⏳ 下载 {name}（{desc}）...")
        tmp = dest.with_suffix(".pt.tmp")
        try:
            with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.rename(dest)
            print(f"✅ {name} 下载完成")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"❌ 下载失败 {url}: {e}\n"
                             f"   可稍后重跑本脚本，或手动执行 sam2_git/checkpoints/download_ckpts.sh")


def print_windows_note() -> None:
    """Windows 下 SAM2_BUILD_CUDA=0 的提示（uv sync 时编译 CUDA 扩展会失败）。"""
    if os.name != "nt":
        return
    print("\n📝 Windows 注意：uv sync 前请确保 SAM2_BUILD_CUDA=0（跳过 .cu 扩展编译），"
          "一次性设置：")
    print('   PowerShell:  setx SAM2_BUILD_CUDA 0   （需重开终端生效）')
    print('   bash:         export SAM2_BUILD_CUDA=0')


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 sam2_git 源码与 checkpoint")
    parser.add_argument("--large-only", action="store_true",
                        help="只下载 large checkpoint（跳过 base_plus）")
    args = parser.parse_args()

    if not shutil.which("git"):
        raise SystemExit("❌ 未找到 git，请先安装 Git")

    clone_or_checkout()
    download_checkpoints(only_large=args.large_only)
    print_windows_note()
    print("\n🎉 初始化完成。下一步：uv sync")


if __name__ == "__main__":
    main()
