"""
PharmaGuard: Intelligent Medication Safety & DDI Prediction System
=================================================================

An AI-powered clinical decision support system for:
- Drug-Drug Interaction (DDI) detection
- Polypharmacy risk assessment
- Drug knowledge graph construction & querying
- Explainable AI for clinical decisions
- Multi-agent clinical decision support

Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "PharmaGuard Team"
__license__ = "MIT"

from pharmaguard.config import Settings

# 延迟导入以避免循环依赖
def get_settings():
    """获取配置"""
    return Settings()