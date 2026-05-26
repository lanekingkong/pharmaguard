"""
PharmaGuard API - FastAPI后端服务
提供药物相互作用预测、风险评估和智能用药安全分析
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import os
import json
from pathlib import Path
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PharmaGuard API",
    description="智能用药安全与DDI预测系统",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================== Pydantic模型 ===================

class DrugInfo(BaseModel):
    """药物信息"""
    name: str = Field(..., description="药物名称")
    dose: Optional[float] = Field(None, description="剂量")
    unit: Optional[str] = Field(None, description="单位")
    route: Optional[str] = Field("口服", description="给药途径")
    frequency: Optional[str] = Field(None, description="给药频率")
    duration: Optional[str] = Field(None, description="用药时长")
    indication: Optional[str] = Field(None, description="适应症")


class PatientInfo(BaseModel):
    """患者信息"""
    patient_id: Optional[str] = Field(None, description="患者ID")
    age: int = Field(..., ge=0, le=150, description="年龄")
    weight: Optional[float] = Field(None, ge=0, description="体重(kg)")
    height: Optional[float] = Field(None, ge=0, description="身高(cm)")
    gender: Optional[str] = Field(None, description="性别")
    primary_diagnosis: Optional[str] = Field(None, description="主要诊断")
    comorbidities: List[str] = Field(default_factory=list, description="合并症")
    allergies: List[str] = Field(default_factory=list, description="过敏史")
    egfr: Optional[float] = Field(None, description="估算肾小球滤过率")
    alt: Optional[float] = Field(None, description="ALT")
    ast: Optional[float] = Field(None, description="AST")
    inr: Optional[float] = Field(None, description="INR")
    genetic_profile: Dict[str, str] = Field(default_factory=dict, description="基因型信息")


class PrescriptionRequest(BaseModel):
    """处方分析请求"""
    patient: PatientInfo = Field(..., description="患者信息")
    medications: List[DrugInfo] = Field(..., description="药物列表")
    
    @validator('medications')
    def validate_medications(cls, v):
        if len(v) < 1:
            raise ValueError('至少需要一种药物')
        if len(v) > 50:
            raise ValueError('最多支持50种药物')
        # 检查重复药物名
        names = [m.name.lower() for m in v]
        if len(names) != len(set(names)):
            raise ValueError('药物列表中存在重复的药物名称')
        return v


class DDIPredictionRequest(BaseModel):
    """DDI预测请求"""
    drug_a: str = Field(..., description="药物A名称")
    drug_b: str = Field(..., description="药物B名称")
    patient_info: Optional[PatientInfo] = Field(None, description="患者信息（可选）")


class DrugQueryRequest(BaseModel):
    """药物查询请求"""
    drug_name: str = Field(..., description="药物名称")


class DrugSimilarityRequest(BaseModel):
    """药物相似度查询请求"""
    drug_a: str = Field(..., description="药物A名称")
    drug_b: str = Field(..., description="药物B名称")


class BatchDDIRequest(BaseModel):
    """批量DDI预测请求"""
    medication_list: List[str] = Field(..., description="药物名称列表")
    patient_info: Optional[PatientInfo] = Field(None, description="患者信息")


class SafetyReportRequest(BaseModel):
    """安全报告请求"""
    patient_id: str = Field(..., description="患者ID")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")
    include_details: bool = Field(True, description="是否包含详细信息")


# =================== 响应模型 ===================

class APIResponse(BaseModel):
    """标准API响应"""
    success: bool
    data: Optional[Any] = None
    message: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = None


class DDIPredictionResponse(BaseModel):
    """DDI预测响应"""
    drug_a: str
    drug_b: str
    has_interaction: bool
    probability: float
    severity: str
    confidence: float
    mechanism: Optional[str] = None
    description: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)


class PrescriptionAnalysisResponse(BaseModel):
    """处方分析响应"""
    patient_summary: Dict[str, Any]
    ddi_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    action_items: List[Dict[str, Any]]
    monitoring_plan: Dict[str, Any]
    patient_education: Dict[str, Any]


# =================== API路由 ===================

@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - API信息"""
    return """
    <html>
    <head><title>PharmaGuard API</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1>PharmaGuard API</h1>
        <p>智能用药安全与DDI预测系统 v1.0.0</p>
        <p>API文档: <a href="/api/docs">/api/docs</a></p>
        <p>状态: 运行中</p>
        <h3>主要端点:</h3>
        <ul>
            <li><code>POST /api/v1/predict/ddi</code> - DDI预测</li>
            <li><code>POST /api/v1/analyze/prescription</code> - 处方分析</li>
            <li><code>POST /api/v1/query/drug</code> - 药物查询</li>
            <li><code>POST /api/v1/query/similarity</code> - 药物相似度</li>
            <li><code>GET /api/v1/stats</code> - 系统统计</li>
            <li><code>GET /api/v1/health</code> - 健康检查</li>
        </ul>
    </body>
    </html>
    """


@app.get("/api/v1/health", response_model=APIResponse)
async def health_check():
    """健康检查"""
    return APIResponse(
        success=True,
        data={
            'status': 'healthy',
            'version': '1.0.0',
            'uptime': 'running',
            'components': {
                'api': 'ok',
                'kg_database': 'checking',
                'ml_models': 'checking'
            }
        },
        message="PharmaGuard API 运行正常"
    )


@app.get("/api/v1/stats", response_model=APIResponse)
async def system_stats():
    """系统统计信息"""
    stats = {
        'ddi_predictions': 0,
        'prescriptions_analyzed': 0,
        'drugs_in_database': 0,
        'interactions_known': 0,
        'system_uptime': 'running'
    }
    
    return APIResponse(
        success=True,
        data=stats,
        message="系统统计信息已获取"
    )


@app.post("/api/v1/predict/ddi", response_model=APIResponse)
async def predict_ddi(request: DDIPredictionRequest):
    """预测药物相互作用"""
    try:
        logger.info(f"DDI预测请求: {request.drug_a} + {request.drug_b}")
        
        # 模拟DDI预测（实际应用中调用模型）
        prediction = {
            'drug_a': request.drug_a,
            'drug_b': request.drug_b,
            'has_interaction': True,
            'probability': 0.85,
            'severity': 'moderate',
            'confidence': 0.78,
            'mechanism': 'CYP3A4代谢竞争',
            'description': f'{request.drug_a}和{request.drug_b}可能通过CYP3A4酶产生代谢竞争',
            'recommendations': [
                '监测药物浓度',
                '必要时调整剂量',
                '观察不良反应'
            ]
        }
        
        return APIResponse(
            success=True,
            data=prediction,
            message=f"DDI预测完成: {request.drug_a} + {request.drug_b}"
        )
        
    except Exception as e:
        logger.error(f"DDI预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/prescription", response_model=APIResponse)
async def analyze_prescription(request: PrescriptionRequest):
    """分析处方安全性"""
    try:
        logger.info(f"处方分析请求: 患者{request.patient.age}岁, {len(request.medications)}种药物")
        
        patient_info = request.patient.model_dump()
        medications = [med.model_dump() for med in request.medications]
        
        # 模拟处方分析（实际应用中调用SafeRx Agent）
        analysis = {
            'patient_summary': {
                'age': patient_info['age'],
                'comorbidities_count': len(patient_info.get('comorbidities', [])),
                'medications_count': len(medications),
                'risk_factors': ['高龄' if patient_info['age'] >= 65 else None,
                               '多重用药' if len(medications) >= 5 else None,
                               '肾功能不全' if patient_info.get('egfr') and patient_info['egfr'] < 60 else None]
            },
            'ddi_analysis': {
                'total_pairs': len(medications) * (len(medications) - 1) // 2,
                'severe_interactions': 0,
                'moderate_interactions': 0,
                'mild_interactions': 0,
                'ddi_details': []
            },
            'risk_assessment': {
                'total_score': 45,
                'max_score': 100,
                'level': 'high',
                'factors': []
            },
            'alerts': [],
            'recommendations': [],
            'action_items': [],
            'monitoring_plan': {
                'items': [],
                'next_review': '建议1周内复查'
            },
            'patient_education': {
                'education_materials': [],
                'safety_tips': []
            }
        }
        
        # 计算DDI对
        for i, med_a in enumerate(medications):
            for j, med_b in enumerate(medications):
                if i >= j:
                    continue
                
                ddi_detail = {
                    'drug_a': med_a['name'],
                    'drug_b': med_b['name'],
                    'severity': 'unknown',
                    'confidence': 0.5,
                    'mechanism': '待分析',
                    'recommendation': '建议进行详细评估'
                }
                
                analysis['ddi_analysis']['ddi_details'].append(ddi_detail)
        
        if patient_info['age'] >= 65:
            analysis['alerts'].append({
                'level': 'warning',
                'title': '老年患者用药风险',
                'description': f'患者年龄{patient_info["age"]}岁，需要关注药物代谢变化'
            })
        
        if len(medications) >= 5:
            analysis['alerts'].append({
                'level': 'warning',
                'title': '多重用药风险',
                'description': f'患者同时使用{len(medications)}种药物'
            })
        
        if patient_info.get('egfr') and patient_info['egfr'] < 60:
            analysis['alerts'].append({
                'level': 'warning',
                'title': '肾功能不全',
                'description': f'eGFR为{patient_info["egfr"]}，需调整药物剂量'
            })
        
        return APIResponse(
            success=True,
            data=analysis,
            message=f"处方分析完成，发现{len(analysis['alerts'])}个风险因素"
        )
        
    except Exception as e:
        logger.error(f"处方分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/drug", response_model=APIResponse)
async def query_drug(request: DrugQueryRequest):
    """查询药物信息"""
    try:
        drug_name = request.drug_name
        
        # 模拟药物查询（实际应用中查询知识图谱）
        drug_info = {
            'name': drug_name,
            'drugbank_id': f'DB{hash(drug_name) % 100000:05d}',
            'atc_codes': [],
            'indications': [],
            'targets': [],
            'side_effects': [],
            'interactions': [],
            'pharmacokinetics': {}
        }
        
        return APIResponse(
            success=True,
            data=drug_info,
            message=f"药物 '{drug_name}' 查询完成"
        )
        
    except Exception as e:
        logger.error(f"药物查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/ddi/batch", response_model=APIResponse)
async def batch_predict_ddi(request: BatchDDIRequest):
    """批量DDI预测"""
    try:
        medications = request.medication_list
        logger.info(f"批量DDI预测: {len(medications)}种药物")
        
        if len(medications) < 2:
            raise HTTPException(status_code=400, detail="至少需要2种药物")
        
        if len(medications) > 30:
            raise HTTPException(status_code=400, detail="最多支持30种药物")
        
        ddi_results = []
        
        for i in range(len(medications)):
            for j in range(i + 1, len(medications)):
                ddi_results.append({
                    'drug_a': medications[i],
                    'drug_b': medications[j],
                    'has_interaction': True,
                    'probability': 0.5 + (hash(f"{medications[i]}_{medications[j]}") % 50) / 100,
                    'severity': 'moderate',
                    'confidence': 0.6 + (hash(f"{medications[j]}_{medications[i]}") % 30) / 100
                })
        
        # 统计
        severe_count = sum(1 for d in ddi_results if d['probability'] > 0.8)
        moderate_count = sum(1 for d in ddi_results if 0.5 < d['probability'] <= 0.8)
        mild_count = sum(1 for d in ddi_results if d['probability'] <= 0.5)
        
        return APIResponse(
            success=True,
            data={
                'total_pairs': len(ddi_results),
                'severe_count': severe_count,
                'moderate_count': moderate_count,
                'mild_count': mild_count,
                'results': ddi_results
            },
            message=f"批量DDI预测完成，共{len(ddi_results)}对药物组合"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量DDI预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/similarity", response_model=APIResponse)
async def query_drug_similarity(request: DrugSimilarityRequest):
    """查询药物相似度"""
    try:
        # 模拟相似度计算
        similarity_result = {
            'drug_a': request.drug_a,
            'drug_b': request.drug_b,
            'similarities': {
                'structural': 0.45,
                'target': 0.30,
                'indication': 0.20,
                'side_effect': 0.15,
                'overall': 0.275
            },
            'common_features': {
                'targets': [],
                'indications': [],
                'side_effects': []
            }
        }
        
        return APIResponse(
            success=True,
            data=similarity_result,
            message=f"药物相似度计算完成"
        )
        
    except Exception as e:
        logger.error(f"相似度计算失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/drugs/search", response_model=APIResponse)
async def search_drugs(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    limit: int = Query(10, ge=1, le=50, description="返回结果数量")
):
    """搜索药物"""
    try:
        # 模拟搜索
        results = [
            {
                'name': f'{q}_result_{i}',
                'drugbank_id': f'DB{i:05d}',
                'match_score': 1.0 - i * 0.1
            }
            for i in range(min(limit, 5))
        ]
        
        return APIResponse(
            success=True,
            data={'results': results, 'total': len(results)},
            message=f"找到 {len(results)} 个相关药物"
        )
        
    except Exception as e:
        logger.error(f"药物搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts/check", response_model=APIResponse)
async def check_alerts(
    patient_id: Optional[str] = Query(None, description="患者ID")
):
    """检查安全警报"""
    alerts = [
        {
            'alert_id': f'ALERT_{i}',
            'level': ['warning', 'critical', 'info'][i % 3],
            'title': f'安全警报 {i+1}',
            'description': '这是一个模拟的安全警报',
            'timestamp': datetime.now().isoformat()
        }
        for i in range(3)
    ]
    
    return APIResponse(
        success=True,
        data={'alerts': alerts, 'total': len(alerts)},
        message=f"获取到 {len(alerts)} 个安全警报"
    )


# =================== 错误处理 ===================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'success': False,
            'message': exc.detail,
            'timestamp': datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            'success': False,
            'message': 'Internal Server Error',
            'timestamp': datetime.now().isoformat()
        }
    )


# =================== 启动配置 ===================

def create_app() -> FastAPI:
    """创建应用实例"""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )