"""
PharmaGuard Configuration Management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, field_validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelDevice(str, Enum):
    """模型设备类型"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """应用配置"""
    
    # =================== 项目配置 ===================
    project_name: str = Field(default="PharmaGuard", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    environment: str = Field(default="development", description="环境: development/production")
    
    # =================== 路径配置 ===================
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent, description="项目根目录")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data", description="数据目录")
    model_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "models", description="模型目录")
    log_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs", description="日志目录")
    
    # =================== Neo4j配置 ===================
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j用户名")
    neo4j_password: str = Field(default="password", description="Neo4j密码")
    neo4j_database: str = Field(default="neo4j", description="Neo4j数据库")
    
    # =================== 模型配置 ===================
    model_device: ModelDevice = Field(default=ModelDevice.CPU, description="模型运行设备")
    model_precision: str = Field(default="float32", description="模型精度")
    model_batch_size: int = Field(default=32, description="批处理大小")
    model_hidden_dim: int = Field(default=256, description="隐藏层维度")
    model_output_dim: int = Field(default=128, description="输出层维度")
    model_dropout: float = Field(default=0.2, description="Dropout率")
    model_learning_rate: float = Field(default=1e-3, description="学习率")
    
    # =================== API配置 ===================
    api_host: str = Field(default="0.0.0.0", description="API主机")
    api_port: int = Field(default=8000, description="API端口")
    api_workers: int = Field(default=4, description="API工作进程数")
    api_timeout: int = Field(default=300, description="API超时时间(秒)")
    api_max_requests: int = Field(default=1000, description="最大请求数")
    api_secret_key: str = Field(default="change-this-to-a-random-secret-key", description="API密钥")
    api_jwt_expiration: int = Field(default=3600, description="JWT过期时间(秒)")
    api_cors_origins: str = Field(default="*", description="CORS允许的源")
    
    # =================== 仪表板配置 ===================
    dashboard_port: int = Field(default=8501, description="仪表板端口")
    dashboard_theme: str = Field(default="light", description="仪表板主题")
    dashboard_host: str = Field(default="0.0.0.0", description="仪表板主机")
    
    # =================== 日志配置 ===================
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    log_format: str = Field(default="json", description="日志格式: json/text")
    log_file: str = Field(default="pharmaguard.log", description="日志文件名")
    log_rotation: str = Field(default="1 day", description="日志轮转周期")
    log_retention: str = Field(default="30 days", description="日志保留时间")
    
    # =================== 外部API配置 ===================
    drugbank_api_key: Optional[str] = Field(default=None, description="DrugBank API密钥")
    pubchem_api_key: Optional[str] = Field(default=None, description="PubChem API密钥")
    openfda_api_key: Optional[str] = Field(default=None, description="OpenFDA API密钥")
    
    # =================== 监控配置 ===================
    enable_metrics: bool = Field(default=True, description="启用监控指标")
    metrics_port: int = Field(default=9090, description="监控指标端口")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    
    # =================== 开发配置 ===================
    debug: bool = Field(default=False, description="调试模式")
    profile: bool = Field(default=False, description="性能分析模式")
    hot_reload: bool = Field(default=False, description="热重载")
    
    # =================== 风险配置 ===================
    ddi_threshold: float = Field(default=0.5, description="DDI检测阈值")
    risk_threshold_high: int = Field(default=60, description="高风险阈值")
    risk_threshold_critical: int = Field(default=80, description="极高风险阈值")
    max_medications: int = Field(default=30, description="最大药物分析数量")
    
    # =================== 验证器 ===================
    @field_validator("base_dir", "data_dir", "model_dir", "log_dir", mode="before")
    @classmethod
    def ensure_directories_exist(cls, v) -> Path:
        """确保目录存在"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("model_device", mode="before")
    @classmethod
    def validate_model_device(cls, v) -> ModelDevice:
        """验证模型设备"""
        if isinstance(v, str):
            v = ModelDevice(v.lower())
        if v == ModelDevice.CUDA:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA不可用，回退到CPU")
                    return ModelDevice.CPU
            except ImportError:
                logger.warning("PyTorch未安装，回退到CPU")
                return ModelDevice.CPU
        elif v == ModelDevice.MPS:
            try:
                import torch
                if not torch.backends.mps.is_available():
                    logger.warning("MPS不可用，回退到CPU")
                    return ModelDevice.CPU
            except ImportError:
                logger.warning("PyTorch未安装，回退到CPU")
                return ModelDevice.CPU
        return v
    
    @field_validator("api_secret_key", mode="before")
    @classmethod
    def validate_secret_key(cls, v) -> str:
        """验证密钥安全性"""
        if v == "change-this-to-a-random-secret-key":
            logger.warning("使用默认密钥，生产环境请修改")
        return v
    
    # =================== 属性方法 ===================
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment.lower() == "production"
    
    @property
    def model_path(self) -> Path:
        """模型路径"""
        return self.model_dir / "ddi_model.pth"
    
    @property
    def knowledge_graph_path(self) -> Path:
        """知识图谱数据路径"""
        return self.data_dir / "knowledge_graph"
    
    @property
    def drug_data_path(self) -> Path:
        """药物数据路径"""
        return self.data_dir / "drugs"
    
    @property
    def log_file_path(self) -> Path:
        """日志文件路径"""
        return self.log_dir / self.log_file
    
    # =================== 配置加载 ===================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            """自定义配置源"""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def reload_settings() -> Settings:
    """重新加载配置"""
    global settings
    settings = Settings()
    return settings


def print_settings_summary() -> None:
    """打印配置摘要"""
    logger.info("=" * 60)
    logger.info("PharmaGuard Configuration Summary")
    logger.info("=" * 60)
    
    summary = {
        "Project": {
            "Name": settings.project_name,
            "Version": settings.version,
            "Environment": settings.environment,
        },
        "Paths": {
            "Base Directory": str(settings.base_dir),
            "Data Directory": str(settings.data_dir),
            "Model Directory": str(settings.model_dir),
            "Log Directory": str(settings.log_dir),
        },
        "Neo4j": {
            "URI": settings.neo4j_uri,
            "User": settings.neo4j_user,
            "Database": settings.neo4j_database,
        },
        "Model": {
            "Device": settings.model_device.value,
            "Precision": settings.model_precision,
            "Batch Size": settings.model_batch_size,
        },
        "API": {
            "Host": settings.api_host,
            "Port": settings.api_port,
            "Workers": settings.api_workers,
        },
        "Dashboard": {
            "Port": settings.dashboard_port,
            "Theme": settings.dashboard_theme,
        },
        "Logging": {
            "Level": settings.log_level.value,
            "Format": settings.log_format,
            "File": str(settings.log_file_path),
        },
    }
    
    for category, configs in summary.items():
        logger.info(f"\n{category}:")
        for key, value in configs.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # 测试配置加载
    print_settings_summary()