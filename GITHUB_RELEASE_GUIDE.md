# PharmaGuard GitHub 发布指南

## 项目概述
**PharmaGuard** 是一个基于人工智能的智能药物安全系统，专门解决临床医学中的药物相互作用（DDI）检测和多重用药风险评估问题。项目结合了图神经网络、知识图谱、可解释AI和多智能体临床决策支持，旨在减少全球每年因药物不良反应导致的77万例伤害。

## 项目亮点
- **解决真实临床痛点**：针对药物相互作用检测准确率低、假阳性率高（90%）的临床问题
- **多模态融合**：分子指纹 + 蛋白靶点 + 通路 + 副作用 + 临床上下文
- **可解释AI**：SHAP解释 + 反事实分析 + 临床报告生成
- **全栈部署**：FastAPI + Streamlit + Neo4j + Docker Compose
- **5W1H架构**：完整遵循What/Why/Who/Where/When/How架构原则

## GitHub 发布步骤

### 1. 创建GitHub仓库
```bash
# 在GitHub网站创建新仓库
# 仓库名: pharmaguard
# 描述: Intelligent Medication Safety & DDI Prediction System
# 许可证: MIT
# 添加README: 否（已有）
```

### 2. 本地初始化Git
```bash
cd D:\marvis_work\github_big_proj4\pharmaguard
git init
git add .
git commit -m "Initial commit: PharmaGuard v1.0.0 - AI-powered drug safety system"
```

### 3. 连接到GitHub远程仓库
```bash
git remote add origin https://github.com/<your-username>/pharmaguard.git
git branch -M main
git push -u origin main
```

### 4. 设置GitHub Actions（可选）
创建 `.github/workflows/ci.yml`：
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

### 5. 添加GitHub徽章（在README.md中）
```markdown
[![CI](https://github.com/your-username/pharmaguard/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/pharmaguard/actions)
[![Codecov](https://codecov.io/gh/your-username/pharmaguard/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/pharmaguard)
[![PyPI](https://img.shields.io/pypi/v/pharmaguard)](https://pypi.org/project/pharmaguard/)
```

### 6. 发布到PyPI（可选）
```bash
pip install build twine
python -m build
twine upload dist/*
```

## 项目结构
```
pharmaguard/
├── src/pharmaguard/              # 核心源代码
│   ├── ddi_detector/            # DDI预测模型（GAT）
│   ├── risk_assessor/           # 多重用药风险评估
│   ├── medkg/                   # 药物知识图谱（Neo4j）
│   ├── xai_explainer/           # 可解释AI（SHAP）
│   ├── saferx_agent/            # 多智能体临床系统
│   ├── api/                     # FastAPI后端
│   ├── dashboard/               # Streamlit仪表板
│   ├── cli.py                   # 命令行工具
│   └── config.py                # 配置管理
├── tests/                       # 单元测试
├── docs/                        # 文档
├── docker/                      # Docker配置
├── data/                        # 数据目录
├── .env.example                 # 环境配置模板
├── docker-compose.yml           # 全栈部署
├── Dockerfile                   # 容器构建
├── pyproject.toml              # 项目元数据
├── setup.py                    # 安装脚本
├── requirements.txt            # 依赖列表
├── LICENSE                     # MIT许可证
└── README.md                   # 完整文档
```

## 快速启动方式
### Docker（推荐）
```bash
git clone https://github.com/your-username/pharmaguard.git
cd pharmaguard
docker compose up -d
# 访问: API (http://localhost:8000/api/docs) | 仪表板 (http://localhost:8501)
```

### 本地安装
```bash
pip install -e ".[dashboard]"
pharmaguard serve --port 8000
pharmaguard dashboard
```

## 临床价值
1. **减少药物伤害**：预测高风险DDI，减少不良反应
2. **优化多重用药**：为老年患者（30-40%使用5+药物）提供风险评估
3. **临床决策支持**：为医生、药师提供可解释的AI建议
4. **研究平台**：为药物安全研究提供开源工具

## 后续发展
1. **数据集成**：连接医院EHR系统
2. **模型优化**：集成更多临床特征
3. **多语言支持**：支持中文、英文等
4. **移动应用**：开发临床移动端应用

## 引用格式
```bibtex
@software{pharmaguard2025,
  author = {PharmaGuard Team},
  title = {PharmaGuard: Intelligent Medication Safety & DDI Prediction System},
  year = {2025},
  url = {https://github.com/pharmaguard/pharmaguard},
  version = {1.0.0}
}
```

## 联系方式
- GitHub Issues: https://github.com/your-username/pharmaguard/issues
- 邮箱: contact@pharmaguard.ai
- 文档: https://pharmaguard.readthedocs.io

---

**PharmaGuard** — 通过AI让多重用药更安全
*"First, do no harm" — now empowered by artificial intelligence.*