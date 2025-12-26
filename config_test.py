# validate_config.py
import yaml
import os
from pathlib import Path


def validate_config():
    """验证配置文件"""
    config_path = Path("config/config.yaml")

    if not config_path.exists():
        print("❌ 配置文件不存在: config/config.yaml")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ 配置文件格式错误: {e}")
            return False

    # 检查必要字段
    required_fields = [
        'llm.api_key',
        'embedding.model',
        'vector_store.collection_name',
        'vector_store.persist_directory'
    ]

    missing_fields = []
    for field_path in required_fields:
        keys = field_path.split('.')
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                missing_fields.append(field_path)
                break

    if missing_fields:
        print(f"❌ 缺少必要配置字段: {missing_fields}")
        return False

    # 验证 API 密钥
    api_key = config.get('llm', {}).get('api_key')
    if not api_key:
        print("❌ API 密钥未配置")
        return False

    # 如果是环境变量格式，检查环境变量
    if isinstance(api_key, str) and api_key.startswith('${') and api_key.endswith('}'):
        env_var = api_key[2:-1]  # 移除 ${ 和 }
        actual_key = os.getenv(env_var)
        if not actual_key:
            print(f"❌ 环境变量 {env_var} 未设置")
            return False
        print(f"✅ 使用环境变量 {env_var} 中的 API 密钥")
    else:
        print("✅ 使用配置文件中的 API 密钥")

    print("✅ 配置验证通过")
    return True


if __name__ == "__main__":
    validate_config()
