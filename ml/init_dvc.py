#!/usr/bin/env python3
"""
初始化DVC
"""
import os
import subprocess
import sys

def run_command(cmd, description):
    """运行命令"""
    print(f"🚀 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e}")
        return False

def main():
    print("🔧 初始化DVC...")
    
    # 创建必要目录
    os.makedirs('.dvc', exist_ok=True)
    os.makedirs('dvc_storage', exist_ok=True)
    
    commands = [
        # 初始化DVC
        ("dvc init", "初始化DVC"),
        
        # 添加远程存储
        ("dvc remote add -d local ./dvc_storage", "设置本地远程存储"),
        
        # 跟踪数据目录
        ("dvc add Corel-1000", "跟踪Corel-1000数据集"),
        
        # 添加文件到git
        ("git add .dvc Corel-1000.dvc .gitignore", "添加DVC文件到Git"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print("❌ DVC初始化失败")
            return
    
    print("""
    🎉 DVC初始化完成！
    
    下一步操作:
    1. 提交到Git: git commit -m "Initialize DVC"
    2. 运行流水线: dvc repro
    3. 推送数据: dvc push
    """)

if __name__ == "__main__":
    main()