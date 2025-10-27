import os
import subprocess

# 初始化DVC
subprocess.run(["dvc", "init"], check=True)

# （可选）配置DVC远程存储（以Azure为例，需替换为你的远程配置）
# subprocess.run(["dvc", "remote", "add", "-d", "azure", "azure://<容器名>"], check=True)
# subprocess.run(["dvc", "remote", "modify", "azure", "connection_string", "<你的Azure连接字符串>"], check=True)

print("DVC初始化完成")