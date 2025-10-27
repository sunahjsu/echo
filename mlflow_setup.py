import os
import sys

def setup_mlflow():
    """设置 MLflow 连接"""
    try:
        # 尝试导入 dagshub
        import dagshub
        import mlflow
        
        print("正在初始化 DagsHub 连接...")
        
        # 设置 DagsHub
        dagshub.init(
            repo_owner='sunahjsu', 
            repo_name='echo', 
            mlflow=True
        )
        
        print("✅ DagsHub 连接成功")
        return True
        
    except ImportError:
        print("❌ dagshub 模块未安装")
        print("请运行: pip install dagshub")
        return False
        
    except Exception as e:
        print(f"❌ DagsHub 连接失败: {e}")
        print("使用本地 MLflow 服务器...")
        
        try:
            import mlflow
            mlflow.set_tracking_uri("http://127.0.0.1:5001")
            mlflow.set_experiment("PCA_Image_Retrieval")
            print("✅ 本地 MLflow 设置成功")
            return True
        except Exception as e2:
            print(f"❌ MLflow 设置失败: {e2}")
            return False

if __name__ == "__main__":
    setup_mlflow()