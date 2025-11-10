# 文件: my_CIL_V1/data_process/KAIST/_init_path.py
import sys
import os

# 获取当前文件所在目录（即 KAIST 文件夹）
project_root = os.path.dirname(os.path.abspath(__file__))

# 添加当前目录（KAIST）
if project_root not in sys.path:
    sys.path.append(project_root)

# 添加 my_lib 目录
lib_path = os.path.join(project_root, "my_lib")
if lib_path not in sys.path:
    sys.path.append(lib_path)

print(f"✅ 已添加路径：{project_root}")
print(f"✅ 已添加 my_lib：{lib_path}")
