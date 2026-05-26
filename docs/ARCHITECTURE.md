# PharmaGuard 系统架构设计

## 1. 总体架构

### 1.1 架构原则

1. **模块化设计**：每个核心功能独立模块，松耦合
2. **可扩展性**：支持新模型、新数据源的轻松集成
3. **可解释性**：所有决策可追溯、可解释
4. **实时性**：API响应时间 < 100ms
5. **安全性**：医疗数据加密、访问控制、审计日志

### 1.2 技术栈

| 组件 | 技术选择 | 理由 |
|------|----------|------|
| **后端框架** | FastAPI | 高性能、异步支持、自动文档生成 |
| **深度学习** | PyTorch | 灵活、研究友好、GNN支持好 |
| **知识图谱** | Neo4j | 图数据库、Cypher查询语言 |
| **关系数据库** | PostgreSQL | ACID、JSON支持、成熟生态 |
| **缓存** | Redis | 高性能、数据结构丰富 |
| **对象存储** | MinIO/S3 | 模型文件、大文件存储 |
| **消息队列** | RabbitMQ | 异步任务、事件驱动 |
| **前端** | React + TypeScript | 组件化、类型安全 |
| **容器** | Docker + Kubernetes | 部署标准化、弹性伸缩 |

### 1.3 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Ingress │ │  API     │ │  Model   │ │  KG      │       │
│  │  Nginx   │ │  Service │ │  Service │ │  Service │       │
│  │          │ │ (FastAPI)│ │(PyTorch) │ │ (Neo4j)  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│         │            │            │            │            │
├─────────┼────────────┼────────────┼────────────┼────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Redis   │ │  Postgres│ │  MinIO   │ │ RabbitMQ │       │
│  │  Cache   │ │  DB      │ │ Storage  │ │  Queue   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 2. 核心模块设计

### 2.1 DDI-Detector 模块

#### 2.1.1 模型架构

```
输入层
├── 药物A特征
│   ├── 分子指纹 (ECFP4, 1024维)
│   ├── 靶点嵌入 (Transformer, 256维)
│   ├── 通路向量 (Pathway2Vec, 128维)
│   └── 副作用向量 (SIDER, 64维)
├── 药物B特征（同上）
└── 上下文特征
    ├── 患者年龄、性别
    ├── 肝肾功能指标
    └── 基因型

融合层
├── GAT层（图注意力网络）
├── Transformer编码器
└── 交叉注意力机制

输出层
├── 二分类：是否有DDI
├── 多分类：DDI严重程度（轻度/中度/重度）
└── 回归：相互作用强度
```

#### 2.1.2 训练流程

```python
# 伪代码
class DDIModel(nn.Module):
    def __init__(self):
        self.drug_encoder = DrugEncoder()
        self.gat = GATConv()
        self.transformer = TransformerEncoder()
        self.classifier = nn.Linear(512, 2)
    
    def forward(self, drug_a, drug_b, context):
        # 药物编码
        a_emb = self.drug_encoder(drug_a)
        b_emb = self.drug_encoder(drug_b)
        
        # 构建药物对图
        graph = build_drug_pair_graph(a_emb, b_emb)
        
        # GAT处理
        gat_out = self.gat(graph)
        
        # Transformer融合上下文
        combined = torch.cat([gat_out, context], dim=-1)
        transformer_out = self.transformer(combined)
        
        # 分类
        output = self.classifier(transformer_out)
        return output
```

### 2.2 Risk-Assessor 模块

#### 2.2.1 风险评估框架

```
风险维度：
├── 药物相关风险
│   ├── DDI风险评分
│   ├── 副作用叠加风险
│   └── 治疗窗狭窄药物
├── 患者相关风险
│   ├── 年龄风险（老年/儿童）
│   ├── 器官功能风险（肝/肾）
│   ├── 基因型风险（CYP代谢）
│   └── 合并症风险
└── 治疗相关风险
    ├── 用药依从性
    ├── 治疗持续时间
    └── 监测需求
```

#### 2.2.2 风险评分算法

```python
class RiskScorer:
    def calculate_polypharmacy_score(self, medications, patient_info):
        # 基础分数
        base_score = len(medications) * 5
        
        # 年龄调整
        if patient_info.age > 65:
            base_score *= 1.3
        
        # 肾功能调整
        if patient_info.egfr < 60:
            base_score *= 1.5
        
        # DDI风险叠加
        ddi_risk = self.calculate_ddi_risk(medications)
        base_score += ddi_risk * 10
        
        # 基因风险
        genetic_risk = self.check_genetic_risks(medications, patient_info.genotype)
        base_score += genetic_risk * 8
        
        return min(base_score, 100)  # 归一化到0-100
```

### 2.3 MedKG 知识图谱

#### 2.3.1 图谱模式

```
节点类型：
├── Drug (药物)
│   ├── 属性：名称、ATC编码、分子式、适应症
│   └── 标签：化学分类、治疗分类
├── Protein (蛋白质/靶点)
│   ├── 属性：UniProt ID、名称、功能
│   └── 标签：酶、受体、转运体
├── Disease (疾病)
│   ├── 属性：ICD编码、名称、描述
│   └── 标签：系统分类
├── Pathway (通路)
│   └── 属性：KEGG ID、Reactome ID
└── Gene (基因)
    └── 属性：HGNC符号、染色体位置

关系类型：
├── INTERACTS_WITH (药物-药物相互作用)
│   └── 属性：严重程度、机制、证据等级
├── TARGETS (药物-靶点作用)
│   └── 属性：作用类型、亲和力
├── INDICATES (药物-疾病适应症)
│   └── 属性：批准状态、证据等级
├── CAUSES (药物-副作用)
│   └── 属性：频率、严重程度
├── PARTICIPATES_IN (靶点-通路参与)
└── ASSOCIATED_WITH (基因-疾病关联)
```

#### 2.3.2 查询示例

```cypher
// 查找药物A的所有潜在DDI
MATCH (d1:Drug {name: 'Warfarin'})-[r:INTERACTS_WITH]-(d2:Drug)
WHERE r.severity = 'severe'
RETURN d2.name AS interacting_drug, 
       r.mechanism AS mechanism,
       r.evidence_level AS evidence

// 查找药物作用通路
MATCH (d:Drug {name: 'Metformin'})-[:TARGETS]->(p:Protein)
      -[:PARTICIPATES_IN]->(path:Pathway)
RETURN path.name AS pathway, 
       COUNT(p) AS target_count
ORDER BY target_count DESC
```

### 2.4 XAI-Explainer 模块

#### 2.4.1 解释方法

```
1. SHAP (SHapley Additive exPlanations)
   ├── 特征重要性排序
   ├── 依赖图分析
   └── 交互效应检测

2. LIME (Local Interpretable Model-agnostic Explanations)
   ├── 局部线性近似
   └── 特征扰动分析

3. GNNExplainer
   ├── 重要子图识别
   ├── 节点重要性
   └── 边重要性

4. Attention Visualization
   ├── 注意力权重热图
   └── 跨头注意力分析
```

#### 2.4.2 临床解释生成

```python
class ClinicalExplainer:
    def generate_explanation(self, prediction, shap_values, patient_info):
        # 提取关键特征
        top_features = self.get_top_features(shap_values, k=5)
        
        # 生成自然语言解释
        explanation = f"基于患者信息（年龄{patient_info.age}岁，"
        
        if 'renal_impairment' in top_features:
            explanation += "肾功能中度受损，"
        
        if 'cyp2c9_variant' in top_features:
            explanation += "携带CYP2C9慢代谢基因型，"
        
        explanation += f"）和药物组合分析，预测DDI风险为{prediction.risk_level}。"
        
        # 添加具体风险因素
        for feature in top_features:
            if feature in self.clinical_feature_map:
                clinical_term = self.clinical_feature_map[feature]
                explanation += f" 主要风险因素：{clinical_term}。"
        
        return explanation
```

### 2.5 SafeRx Agent 模块

#### 2.5.1 Agent架构

```
SafeRx Agent
├── 感知层 (Perception)
│   ├── 患者信息解析器
│   ├── 处方解析器
│   └── 临床指南检索器
├── 推理层 (Reasoning)
│   ├── 风险推理引擎
│   ├── 治疗建议生成器
│   └── 备选方案评估器
├── 决策层 (Decision)
│   ├── 风险-获益权衡
│   ├── 个性化建议生成
│   └── 优先级排序
└── 执行层 (Action)
    ├── 自然语言生成
    ├── 可视化报告
    └── API响应
```

#### 2.5.2 工作流程

```python
class SafeRxAgent:
    async def process_prescription(self, prescription, patient_info):
        # 1. 信息提取
        extracted_info = await self.extractor.extract(prescription)
        
        # 2. DDI检测
        ddi_results = await self.ddi_detector.predict(extracted_info.drugs)
        
        # 3. 风险评估
        risk_score = await self.risk_assessor.calculate(
            extracted_info.drugs, patient_info
        )
        
        # 4. 知识图谱查询
        kg_insights = await self.medkg.query_relevant_info(
            extracted_info.drugs, patient_info.conditions
        )
        
        # 5. LLM生成建议
        suggestions = await self.llm_generator.generate(
            ddi_results, risk_score, kg_insights, patient_info
        )
        
        # 6. 可解释性增强
        explanations = await self.explainer.explain(
            ddi_results, risk_score, suggestions
        )
        
        return {
            "risk_assessment": risk_score,
            "ddi_warnings": ddi_results,
            "clinical_suggestions": suggestions,
            "explanations": explanations
        }
```

## 3. API设计

### 3.1 RESTful API端点

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|------|
| `/api/v1/ddi/predict` | POST | DDI预测 | JWT |
| `/api/v1/risk/assess` | POST | 风险评估 | JWT |
| `/api/v1/kg/query` | POST | 知识图谱查询 | JWT |
| `/api/v1/agent/consult` | POST | 智能体咨询 | JWT |
| `/api/v1/explanations/generate` | POST | 生成解释 | JWT |
| `/api/v1/models/retrain` | POST | 模型重训练 | Admin |
| `/api/v1/data/upload` | POST | 数据上传 | Admin |

### 3.2 请求/响应示例

```json
// DDI预测请求
{
  "drugs": [
    {
      "name": "warfarin",
      "dose": "5mg",
      "frequency": "once daily"
    },
    {
      "name": "aspirin",
      "dose": "100mg",
      "frequency": "once daily"
    }
  ],
  "patient_info": {
    "age": 68,
    "weight": 75,
    "renal_function": "moderate_impairment",
    "genotype": {
      "cyp2c9": "*2/*3",
      "vkorc1": "GG"
    }
  }
}

// DDI预测响应
{
  "prediction": {
    "has_interaction": true,
    "severity": "severe",
    "confidence": 0.92,
    "mechanism": "Pharmacokinetic - CYP2C9 inhibition"
  },
  "risk_assessment": {
    "overall_risk": "high",
    "risk_score": 85,
    "risk_factors": [
      "Age > 65",
      "Renal impairment",
      "CYP2C9 poor metabolizer"
    ]
  },
  "recommendations": [
    {
      "type": "monitoring",
      "action": "Monitor INR closely",
      "frequency": "Daily for first week"
    },
    {
      "type": "dose_adjustment",
      "action": "Reduce warfarin dose by 30%",
      "rationale": "Increased bleeding risk"
    }
  ],
  "explanations": {
    "shap_values": {...},
    "clinical_explanation": "患者年龄较大且肾功能受损..."
  }
}
```

## 4. 数据流设计

### 4.1 实时预测流程

```
1. 客户端请求 → API网关
2. API网关 → 认证/鉴权
3. 请求分发 → 相应微服务
4. 微服务处理：
   a. 数据验证和预处理
   b. 特征提取和编码
   c. 模型推理
   d. 结果后处理
5. 结果聚合 → API网关
6. API网关 → 客户端响应
```

### 4.2 批量处理流程

```
1. 数据上传 → 消息队列
2. 任务调度器 → 分配处理节点
3. 并行处理：
   a. 数据清洗和标准化
   b. 批量预测
   c. 结果汇总
4. 结果存储 → 数据库/文件系统
5. 通知生成 → 用户/系统
```

## 5. 安全设计

### 5.1 数据安全

- **加密存储**：所有PHI数据AES-256加密
- **传输安全**：TLS 1.3+加密传输
- **数据脱敏**：训练数据脱敏处理
- **访问日志**：完整审计日志

### 5.2 访问控制

- **RBAC模型**：角色-权限-资源
- **JWT认证**：短期访问令牌
- **API限流**：防DDoS攻击
- **IP白名单**：生产环境限制

### 5.3 合规性

- **HIPAA合规**：医疗数据保护
- **GDPR合规**：欧盟数据保护
- **伦理审查**：研究伦理委员会批准
- **透明度**：算法可解释性报告

## 6. 监控与运维

### 6.1 监控指标

- **系统性能**：CPU、内存、磁盘、网络
- **API性能**：响应时间、错误率、吞吐量
- **模型性能**：预测准确率、延迟、漂移检测
- **业务指标**：用户数、请求数、风险预警数

### 6.2 告警策略

- **紧急告警**：系统宕机、数据泄露
- **重要告警**：性能下降、模型漂移
- **警告告警**：资源使用率高、错误率上升
- **信息通知**：日常运维、版本更新

## 7. 扩展性设计

### 7.1 水平扩展

- **无状态服务**：API服务无状态设计
- **负载均衡**：Nginx/K8s Ingress
- **数据库分片**：按患者ID分片
- **缓存集群**：Redis集群

### 7.2 功能扩展

- **插件架构**：新模型插件化集成
- **配置驱动**：功能开关配置化
- **API版本**：向后兼容的API版本
- **数据管道**：可扩展的ETL管道

---

*本架构设计文档将持续更新，反映系统演进。*