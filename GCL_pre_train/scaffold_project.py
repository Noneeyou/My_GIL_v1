#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a machine-learning project scaffold like:

Project_name/
    config/
        config.yaml
    data/
        raw/
        processed/
    notebook/
    src/
        data_utils.py
        eval.py
        models.py
        train.py
    README.md
"""

from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path
import datetime as _dt

README_TEMPLATE = """# {proj}

项目目录结构：

{proj}
├─ config/
│  └─ config.yaml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebook/
├─ src/
│  ├─ data_utils.py
│  ├─ eval.py
│  ├─ models.py
│  └─ train.py
└─ README.md

"""

CONFIG_YAML = """# 生成时间: {now}
project_name: "{proj}"
"""

# 创建文件
def create_file(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"⚠️  Skip (exists): {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"✅ Created: {path}")

# 合法性检查
def is_valid_name(name: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_]+", name) is not None

# 脚手架函数
def scaffold(base: Path, overwrite: bool):
    # 创建目录
    (base / "config").mkdir(parents=True, exist_ok=True)
    (base / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (base / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base / "notebook").mkdir(parents=True, exist_ok=True)
    (base / "src").mkdir(parents=True, exist_ok=True)

    # 创建文件
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    create_file(base / "README.md", README_TEMPLATE.format(proj=base.name), overwrite)
    create_file(base / "config" / "config.yaml", CONFIG_YAML.format(now=now, proj=base.name), overwrite)
    create_file(base / "src" / "data_utils.py", "", overwrite)
    create_file(base / "src" / "eval.py", "", overwrite)
    create_file(base / "src" / "models.py", "", overwrite)
    create_file(base / "src" / "train.py", "", overwrite)

# 解析命令行参数
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Create a standard ML project scaffold.")
    p.add_argument(
        "--name",
        "-n",
        type=str,
        help="项目名称（只能包含英文、数字和下划线）",
    )
    p.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="若目标文件已存在则覆盖",
    )
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    # 如果没给参数，就交互式输入
    project_name = args.name or input("请输入项目名称（英文/数字/下划线）：").strip()

    # 合法性检查
    if not is_valid_name(project_name):
        print("❌ 项目名称不合法！只能包含英文、数字和下划线，不能有中文。")
        sys.exit(1)

    base = Path(project_name).resolve()
    base.mkdir(parents=True, exist_ok=True)
    print(f"📁 Project root: {base}")

    scaffold(base, overwrite=args.overwrite)
    print("🎉 完成！结构已生成。")

if __name__ == "__main__":
    sys.exit(main())
