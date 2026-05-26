"""
SafeRx Agent - 智能用药安全代理
基于多Agent协作的临床决策支持系统
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """代理角色"""
    COORDINATOR = "coordinator"  # 协调者
    DDI_ANALYST = "ddi_analyst"  # DDI分析师
    RISK_ASSESSOR = "risk_assessor"  # 风险评估师
    KNOWLEDGE_EXPERT = "knowledge_expert"  # 知识专家
    CLINICAL_ADVISOR = "clinical_advisor"  # 临床顾问
    PATIENT_EDUCATOR = "patient_educator"  # 患者教育者


class AlertLevel(Enum):
    """警报级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ClinicalContext:
    """临床上下文"""
    patient_id: str
    age: int
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    comorbidities: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    current_medications: List[Dict[str, Any]] = field(default_factory=list)
    lab_results: Dict[str, Any] = field(default_factory=dict)
    genetic_profile: Dict[str, str] = field(default_factory=dict)
    vital_signs: Dict[str, Any] = field(default_factory=dict)
    special_conditions: List[str] = field(default_factory=list)  # 怀孕、哺乳等


@dataclass
class MedicationOrder:
    """药物医嘱"""
    drug_name: str
    dose: float
    unit: str
    route: str
    frequency: str
    duration: Optional[str] = None
    start_date: Optional[str] = None
    indication: Optional[str] = None
    prescriber: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class SafetyAlert:
    """安全警报"""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    drugs_involved: List[str]
    recommendation: str
    evidence_level: str
    references: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClinicalRecommendation:
    """临床建议"""
    recommendation_id: str
    category: str  # ddi, dosing, monitoring, alternative
    priority: int  # 1-5, 1最高
    description: str
    rationale: str
    expected_outcome: str
    alternatives: List[str] = field(default_factory=list)
    monitoring_plan: Optional[str] = None
    follow_up: Optional[str] = None


class BaseAgent:
    """基础代理类"""
    
    def __init__(self, role: AgentRole, name: str):
        self.role = role
        self.name = name
        self.knowledge_base = {}
        self.conversation_history = []
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        raise NotImplementedError
    
    def log_action(self, action: str, details: Dict[str, Any]):
        """记录操作"""
        logger.info(f"[{self.name}] {action}: {json.dumps(details, ensure_ascii=False)}")
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })


class CoordinatorAgent(BaseAgent):
    """协调者代理 - 负责协调其他代理"""
    
    def __init__(self):
        super().__init__(AgentRole.COORDINATOR, "Coordinator")
        self.sub_agents = {}
        
    def register_agent(self, agent: BaseAgent):
        """注册子代理"""
        self.sub_agents[agent.role.value] = agent
        logger.info(f"注册代理: {agent.name} ({agent.role.value})")
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """协调处理流程"""
        self.log_action("开始协调处理", {"context_keys": list(context.keys())})
        
        results = {}
        alerts = []
        recommendations = []
        
        # 1. DDI分析
        if 'ddi_analyst' in self.sub_agents:
            ddi_result = await self.sub_agents['ddi_analyst'].process(context)
            results['ddi_analysis'] = ddi_result
            alerts.extend(ddi_result.get('alerts', []))
            recommendations.extend(ddi_result.get('recommendations', []))
        
        # 2. 风险评估
        if 'risk_assessor' in self.sub_agents:
            risk_result = await self.sub_agents['risk_assessor'].process(context)
            results['risk_assessment'] = risk_result
            alerts.extend(risk_result.get('alerts', []))
            recommendations.extend(risk_result.get('recommendations', []))
        
        # 3. 知识查询
        if 'knowledge_expert' in self.sub_agents:
            knowledge_result = await self.sub_agents['knowledge_expert'].process(context)
            results['knowledge'] = knowledge_result
        
        # 4. 临床建议
        if 'clinical_advisor' in self.sub_agents:
            clinical_result = await self.sub_agents['clinical_advisor'].process({
                **context,
                'ddi_results': results.get('ddi_analysis'),
                'risk_results': results.get('risk_assessment'),
                'knowledge_results': results.get('knowledge')
            })
            results['clinical_advice'] = clinical_result
            recommendations.extend(clinical_result.get('recommendations', []))
        
        # 5. 患者教育
        if 'patient_educator' in self.sub_agents:
            education_result = await self.sub_agents['patient_educator'].process({
                **context,
                'alerts': alerts,
                'recommendations': recommendations
            })
            results['patient_education'] = education_result
        
        # 汇总结果
        final_result = self._synthesize_results(results, alerts, recommendations)
        
        self.log_action("处理完成", {
            "alerts_count": len(alerts),
            "recommendations_count": len(recommendations)
        })
        
        return final_result
    
    def _synthesize_results(
        self,
        results: Dict[str, Any],
        alerts: List[SafetyAlert],
        recommendations: List[ClinicalRecommendation]
    ) -> Dict[str, Any]:
        """综合所有结果"""
        # 按优先级排序建议
        recommendations.sort(key=lambda x: x.priority)
        
        # 按严重程度排序警报
        alert_order = {
            AlertLevel.EMERGENCY: 0,
            AlertLevel.CRITICAL: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3
        }
        alerts.sort(key=lambda x: alert_order.get(x.level, 99))
        
        # 生成综合报告
        synthesis = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(alerts, recommendations),
            'alerts': [self._alert_to_dict(alert) for alert in alerts],
            'recommendations': [self._recommendation_to_dict(rec) for rec in recommendations],
            'detailed_results': results,
            'action_items': self._generate_action_items(alerts, recommendations),
            'monitoring_plan': self._generate_monitoring_plan(recommendations)
        }
        
        return synthesis
    
    def _generate_summary(
        self,
        alerts: List[SafetyAlert],
        recommendations: List[ClinicalRecommendation]
    ) -> str:
        """生成总结"""
        critical_count = sum(1 for a in alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY])
        warning_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)
        
        summary_parts = []
        
        if critical_count > 0:
            summary_parts.append(f"发现 {critical_count} 个严重安全警报，需要立即处理。")
        
        if warning_count > 0:
            summary_parts.append(f"发现 {warning_count} 个警告级别的安全问题。")
        
        if recommendations:
            summary_parts.append(f"共生成 {len(recommendations)} 条临床建议。")
        
        if not summary_parts:
            summary_parts.append("未发现重大安全问题，当前用药方案风险较低。")
        
        return " ".join(summary_parts)
    
    def _alert_to_dict(self, alert: SafetyAlert) -> Dict[str, Any]:
        """转换警报为字典"""
        return {
            'alert_id': alert.alert_id,
            'level': alert.level.value,
            'title': alert.title,
            'description': alert.description,
            'drugs_involved': alert.drugs_involved,
            'recommendation': alert.recommendation,
            'evidence_level': alert.evidence_level,
            'references': alert.references,
            'timestamp': alert.timestamp
        }
    
    def _recommendation_to_dict(self, rec: ClinicalRecommendation) -> Dict[str, Any]:
        """转换建议为字典"""
        return {
            'recommendation_id': rec.recommendation_id,
            'category': rec.category,
            'priority': rec.priority,
            'description': rec.description,
            'rationale': rec.rationale,
            'expected_outcome': rec.expected_outcome,
            'alternatives': rec.alternatives,
            'monitoring_plan': rec.monitoring_plan,
            'follow_up': rec.follow_up
        }
    
    def _generate_action_items(
        self,
        alerts: List[SafetyAlert],
        recommendations: List[ClinicalRecommendation]
    ) -> List[Dict[str, Any]]:
        """生成行动项"""
        action_items = []
        
        # 从警报生成行动项
        for alert in alerts:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                action_items.append({
                    'priority': 'immediate',
                    'action': alert.recommendation,
                    'reason': alert.title,
                    'deadline': '立即'
                })
            elif alert.level == AlertLevel.WARNING:
                action_items.append({
                    'priority': 'high',
                    'action': alert.recommendation,
                    'reason': alert.title,
                    'deadline': '24小时内'
                })
        
        # 从建议生成行动项
        for rec in recommendations:
            if rec.priority <= 2:
                action_items.append({
                    'priority': 'high',
                    'action': rec.description,
                    'reason': rec.rationale,
                    'deadline': '下次用药前'
                })
            elif rec.priority <= 3:
                action_items.append({
                    'priority': 'medium',
                    'action': rec.description,
                    'reason': rec.rationale,
                    'deadline': '一周内'
                })
        
        return action_items
    
    def _generate_monitoring_plan(
        self,
        recommendations: List[ClinicalRecommendation]
    ) -> Dict[str, Any]:
        """生成监测计划"""
        monitoring_items = []
        
        for rec in recommendations:
            if rec.monitoring_plan:
                monitoring_items.append({
                    'item': rec.description,
                    'plan': rec.monitoring_plan,
                    'frequency': rec.follow_up or '按需'
                })
        
        return {
            'items': monitoring_items,
            'next_review': '建议1周内复查',
            'alert_thresholds': {
                'renal_function': 'eGFR下降>25%',
                'hepatic_function': 'ALT/AST升高>3倍正常上限',
                'bleeding_risk': 'INR>4.0或出现出血症状'
            }
        }


class DDIAnalystAgent(BaseAgent):
    """DDI分析代理"""
    
    def __init__(self, ddi_model=None, kg_query=None):
        super().__init__(AgentRole.DDI_ANALYST, "DDI Analyst")
        self.ddi_model = ddi_model
        self.kg_query = kg_query
        
        # DDI严重程度分级标准
        self.severity_criteria = {
            'contraindicated': {
                'level': AlertLevel.EMERGENCY,
                'description': '禁止联合使用',
                'action': '立即停止联合用药'
            },
            'severe': {
                'level': AlertLevel.CRITICAL,
                'description': '严重相互作用，可能危及生命',
                'action': '避免联合使用，如必须使用需严密监测'
            },
            'moderate': {
                'level': AlertLevel.WARNING,
                'description': '中度相互作用，需要调整剂量或监测',
                'action': '调整剂量并加强监测'
            },
            'mild': {
                'level': AlertLevel.INFO,
                'description': '轻度相互作用，通常不需要干预',
                'action': '注意观察，无需特殊处理'
            }
        }
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析DDI"""
        self.log_action("开始DDI分析", {"medications_count": len(context.get('medications', []))})
        
        medications = context.get('medications', [])
        patient_info = context.get('patient_info', {})
        
        alerts = []
        recommendations = []
        ddi_pairs = []
        
        # 分析所有药物对
        for i, med_a in enumerate(medications):
            for j, med_b in enumerate(medications):
                if i >= j:
                    continue
                
                # 查询知识图谱
                ddi_info = None
                if self.kg_query:
                    ddi_info = self.kg_query.find_ddi_by_drugs(
                        med_a.get('name', ''),
                        med_b.get('name', '')
                    )
                
                # 使用模型预测
                model_prediction = None
                if self.ddi_model:
                    model_prediction = self._predict_ddi(med_a, med_b, patient_info)
                
                # 综合判断
                ddi_result = self._evaluate_ddi(med_a, med_b, ddi_info, model_prediction)
                
                if ddi_result:
                    ddi_pairs.append(ddi_result)
                    
                    # 生成警报
                    if ddi_result['severity'] in ['contraindicated', 'severe', 'moderate']:
                        alert = self._create_ddi_alert(med_a, med_b, ddi_result)
                        alerts.append(alert)
                    
                    # 生成建议
                    if ddi_result['severity'] in ['contraindicated', 'severe']:
                        rec = self._create_ddi_recommendation(med_a, med_b, ddi_result)
                        recommendations.append(rec)
        
        # 分析多重DDI（3种及以上药物）
        if len(medications) >= 3:
            multi_ddi_alerts = self._analyze_multi_ddi(medications, ddi_pairs)
            alerts.extend(multi_ddi_alerts)
        
        return {
            'ddi_pairs': ddi_pairs,
            'alerts': alerts,
            'recommendations': recommendations,
            'summary': self._generate_ddi_summary(ddi_pairs, alerts)
        }
    
    def _predict_ddi(
        self,
        med_a: Dict[str, Any],
        med_b: Dict[str, Any],
        patient_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """使用模型预测DDI"""
        # 简化实现：返回模拟预测结果
        # 实际应用中应调用训练好的DDI模型
        return {
            'has_interaction': True,
            'probability': 0.85,
            'severity': 'moderate',
            'confidence': 0.78
        }
    
    def _evaluate_ddi(
        self,
        med_a: Dict[str, Any],
        med_b: Dict[str, Any],
        kg_info: Optional[Dict[str, Any]],
        model_prediction: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """综合评估DDI"""
        severity = 'unknown'
        confidence = 0.0
        mechanism = None
        description = None
        
        # 优先使用知识图谱信息
        if kg_info and kg_info.get('interaction'):
            interaction = kg_info['interaction']
            severity = interaction.get('severity', 'unknown')
            mechanism = interaction.get('mechanism')
            description = interaction.get('description')
            confidence = 0.9  # 知识图谱信息置信度高
        
        # 补充模型预测
        if model_prediction and model_prediction.get('has_interaction'):
            if severity == 'unknown':
                severity = model_prediction.get('severity', 'unknown')
            confidence = max(confidence, model_prediction.get('confidence', 0.0))
        
        if severity == 'unknown' and not model_prediction:
            return None
        
        return {
            'drug_a': med_a.get('name', ''),
            'drug_b': med_b.get('name', ''),
            'severity': severity,
            'confidence': confidence,
            'mechanism': mechanism,
            'description': description,
            'source': 'kg' if kg_info else 'model' if model_prediction else 'unknown'
        }
    
    def _create_ddi_alert(
        self,
        med_a: Dict[str, Any],
        med_b: Dict[str, Any],
        ddi_result: Dict[str, Any]
    ) -> SafetyAlert:
        """创建DDI警报"""
        severity = ddi_result['severity']
        criteria = self.severity_criteria.get(severity, self.severity_criteria['mild'])
        
        return SafetyAlert(
            alert_id=f"DDI_{med_a.get('name')}_{med_b.get('name')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            level=criteria['level'],
            title=f"药物相互作用: {med_a.get('name')} + {med_b.get('name')}",
            description=f"{criteria['description']}: {ddi_result.get('description', '')}",
            drugs_involved=[med_a.get('name', ''), med_b.get('name', '')],
            recommendation=criteria['action'],
            evidence_level='high' if ddi_result.get('source') == 'kg' else 'medium'
        )
    
    def _create_ddi_recommendation(
        self,
        med_a: Dict[str, Any],
        med_b: Dict[str, Any],
        ddi_result: Dict[str, Any]
    ) -> ClinicalRecommendation:
        """创建DDI建议"""
        return ClinicalRecommendation(
            recommendation_id=f"REC_DDI_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            category='ddi',
            priority=1 if ddi_result['severity'] == 'contraindicated' else 2,
            description=f"调整 {med_a.get('name')} 和 {med_b.get('name')} 的联合用药方案",
            rationale=f"存在{ddi_result['severity']}程度的药物相互作用",
            expected_outcome="降低药物不良事件风险",
            alternatives=[f"考虑使用替代药物", f"调整给药时间间隔", f"监测相关指标"],
            monitoring_plan=f"监测{ddi_result.get('mechanism', '相关')}指标",
            follow_up="1周内复查"
        )
    
    def _analyze_multi_ddi(
        self,
        medications: List[Dict[str, Any]],
        ddi_pairs: List[Dict[str, Any]]
    ) -> List[SafetyAlert]:
        """分析多重DDI"""
        alerts = []
        
        # 检查是否有3种及以上药物同时存在相互作用
        severe_pairs = [pair for pair in ddi_pairs if pair['severity'] in ['severe', 'contraindicated']]
        
        if len(severe_pairs) >= 2:
            involved_drugs = set()
            for pair in severe_pairs:
                involved_drugs.add(pair['drug_a'])
                involved_drugs.add(pair['drug_b'])
            
            if len(involved_drugs) >= 3:
                alerts.append(SafetyAlert(
                    alert_id=f"MULTI_DDI_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    title="多重药物相互作用风险",
                    description=f"发现 {len(involved_drugs)} 种药物之间存在 {len(severe_pairs)} 个严重相互作用",
                    drugs_involved=list(involved_drugs),
                    recommendation="建议进行全面的用药重整，由临床药师参与评估",
                    evidence_level='high'
                ))
        
        return alerts
    
    def _generate_ddi_summary(
        self,
        ddi_pairs: List[Dict[str, Any]],
        alerts: List[SafetyAlert]
    ) -> str:
        """生成DDI分析总结"""
        if not ddi_pairs:
            return "未发现药物相互作用。"
        
        severe_count = sum(1 for ddi in ddi_pairs if ddi['severity'] in ['severe', 'contraindicated'])
        moderate_count = sum(1 for ddi in ddi_pairs if ddi['severity'] == 'moderate')
        mild_count = sum(1 for ddi in ddi_pairs if ddi['severity'] == 'mild')
        
        parts = [f"共分析 {len(ddi_pairs)} 对药物组合。"]
        
        if severe_count > 0:
            parts.append(f"发现 {severe_count} 个严重相互作用，需要立即处理。")
        if moderate_count > 0:
            parts.append(f"发现 {moderate_count} 个中度相互作用。")
        if mild_count > 0:
            parts.append(f"发现 {mild_count} 个轻度相互作用。")
        
        return " ".join(parts)


class RiskAssessorAgent(BaseAgent):
    """风险评估代理"""
    
    def __init__(self, risk_scorer=None):
        super().__init__(AgentRole.RISK_ASSESSOR, "Risk Assessor")
        self.risk_scorer = risk_scorer
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估风险"""
        self.log_action("开始风险评估", {})
        
        patient_info = context.get('patient_info', {})
        medications = context.get('medications', [])
        
        alerts = []
        recommendations = []
        
        # 评估年龄风险
        age = patient_info.get('age', 0)
        if age >= 75:
            alerts.append(SafetyAlert(
                alert_id=f"AGE_RISK_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                level=AlertLevel.WARNING,
                title="高龄患者用药风险",
                description=f"患者年龄 {age} 岁，属于高龄患者，药物代谢能力下降",
                drugs_involved=[],
                recommendation="建议根据肾功能调整药物剂量，加强用药监测",
                evidence_level='high'
            ))
        
        # 评估多重用药风险
        if len(medications) >= 5:
            alerts.append(SafetyAlert(
                alert_id=f"POLYPHARMACY_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                level=AlertLevel.WARNING,
                title="多重用药风险",
                description=f"患者同时使用 {len(medications)} 种药物，存在多重用药风险",
                drugs_involved=[med.get('name', '') for med in medications],
                recommendation="建议进行用药重整，评估每种药物的必要性",
                evidence_level='high'
            ))
        
        # 评估肾功能
        egfr = patient_info.get('egfr')
        if egfr is not None and egfr < 60:
            level = AlertLevel.CRITICAL if egfr < 30 else AlertLevel.WARNING
            alerts.append(SafetyAlert(
                alert_id=f"RENAL_RISK_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                level=level,
                title="肾功能不全用药风险",
                description=f"患者eGFR为 {egfr} mL/min/1.73m²，肾功能受损",
                drugs_involved=[],
                recommendation="根据肾功能调整经肾排泄药物的剂量",
                evidence_level='high'
            ))
            
            recommendations.append(ClinicalRecommendation(
                recommendation_id=f"REC_RENAL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                category='dosing',
                priority=1,
                description="根据eGFR调整药物剂量",
                rationale=f"患者eGFR为{egfr}，需要调整经肾排泄药物的剂量",
                expected_outcome="避免药物蓄积和肾毒性",
                monitoring_plan="每1-3个月监测肾功能",
                follow_up="1个月后复查肾功能"
            ))
        
        # 评估肝功能
        alt = patient_info.get('alt')
        ast = patient_info.get('ast')
        if (alt and alt > 120) or (ast and ast > 120):
            alerts.append(SafetyAlert(
                alert_id=f"HEPATIC_RISK_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                level=AlertLevel.WARNING,
                title="肝功能异常用药风险",
                description=f"患者肝功能指标异常（ALT: {alt}, AST: {ast}）",
                drugs_involved=[],
                recommendation="避免使用肝毒性药物，调整经肝代谢药物的剂量",
                evidence_level='high'
            ))
        
        # 计算综合风险评分
        risk_score = self._calculate_risk_score(patient_info, medications)
        
        return {
            'risk_score': risk_score,
            'alerts': alerts,
            'recommendations': recommendations,
            'summary': self._generate_risk_summary(risk_score, alerts)
        }
    
    def _calculate_risk_score(
        self,
        patient_info: Dict[str, Any],
        medications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算风险评分"""
        score = 0
        factors = []
        
        # 年龄因素
        age = patient_info.get('age', 0)
        if age >= 75:
            score += 20
            factors.append({'factor': 'age', 'score': 20, 'description': '高龄患者'})
        elif age >= 65:
            score += 10
            factors.append({'factor': 'age', 'score': 10, 'description': '老年患者'})
        
        # 药物数量
        num_meds = len(medications)
        if num_meds >= 10:
            score += 25
            factors.append({'factor': 'polypharmacy', 'score': 25, 'description': '严重多重用药'})
        elif num_meds >= 7:
            score += 15
            factors.append({'factor': 'polypharmacy', 'score': 15, 'description': '中度多重用药'})
        elif num_meds >= 5:
            score += 8
            factors.append({'factor': 'polypharmacy', 'score': 8, 'description': '轻度多重用药'})
        
        # 肾功能
        egfr = patient_info.get('egfr')
        if egfr is not None:
            if egfr < 30:
                score += 25
                factors.append({'factor': 'renal', 'score': 25, 'description': '严重肾功能不全'})
            elif egfr < 60:
                score += 15
                factors.append({'factor': 'renal', 'score': 15, 'description': '中度肾功能不全'})
        
        # 合并症
        comorbidities = patient_info.get('comorbidities', [])
        if len(comorbidities) >= 3:
            score += 15
            factors.append({'factor': 'comorbidities', 'score': 15, 'description': '多种合并症'})
        elif len(comorbidities) >= 1:
            score += 5
            factors.append({'factor': 'comorbidities', 'score': 5, 'description': '存在合并症'})
        
        # 确定风险等级
        if score >= 60:
            level = 'critical'
        elif score >= 40:
            level = 'high'
        elif score >= 20:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'total_score': score,
            'max_score': 100,
            'level': level,
            'factors': factors
        }
    
    def _generate_risk_summary(
        self,
        risk_score: Dict[str, Any],
        alerts: List[SafetyAlert]
    ) -> str:
        """生成风险总结"""
        level_text = {
            'critical': '极高',
            'high': '高',
            'moderate': '中等',
            'low': '低'
        }
        
        return (
            f"患者用药风险评分为 {risk_score['total_score']}/100，"
            f"属于{level_text.get(risk_score['level'], '未知')}风险级别。"
            f"共发现 {len(alerts)} 个风险因素。"
        )


class ClinicalAdvisorAgent(BaseAgent):
    """临床顾问代理"""
    
    def __init__(self):
        super().__init__(AgentRole.CLINICAL_ADVISOR, "Clinical Advisor")
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提供临床建议"""
        self.log_action("生成临床建议", {})
        
        ddi_results = context.get('ddi_results', {})
        risk_results = context.get('risk_results', {})
        
        recommendations = []
        
        # 基于DDI结果的建议
        if ddi_results:
            ddi_pairs = ddi_results.get('ddi_pairs', [])
            severe_ddis = [ddi for ddi in ddi_pairs if ddi['severity'] in ['severe', 'contraindicated']]
            
            if severe_ddis:
                recommendations.append(ClinicalRecommendation(
                    recommendation_id=f"REC_CLINICAL_{datetime.now().strftime('%Y%m%d%H%M%S')}_1",
                    category='medication_review',
                    priority=1,
                    description="进行全面的用药重整",
                    rationale=f"存在 {len(severe_ddis)} 个严重药物相互作用",
                    expected_outcome="消除或减少严重DDI风险",
                    alternatives=[
                        "停用非必需药物",
                        "替换为相互作用风险较低的药物",
                        "调整给药时间间隔"
                    ],
                    monitoring_plan="用药重整后1周内随访",
                    follow_up="1周后评估"
                ))
        
        # 基于风险评估的建议
        if risk_results:
            risk_score = risk_results.get('risk_score', {})
            if risk_score.get('level') in ['critical', 'high']:
                recommendations.append(ClinicalRecommendation(
                    recommendation_id=f"REC_CLINICAL_{datetime.now().strftime('%Y%m%d%H%M%S')}_2",
                    category='monitoring',
                    priority=2,
                    description="加强临床监测",
                    rationale=f"患者用药风险等级为{risk_score.get('level')}",
                    expected_outcome="及时发现和处理药物不良事件",
                    monitoring_plan="每周监测生命体征和实验室指标",
                    follow_up="根据监测结果调整方案"
                ))
        
        # 通用建议
        recommendations.append(ClinicalRecommendation(
            recommendation_id=f"REC_CLINICAL_{datetime.now().strftime('%Y%m%d%H%M%S')}_3",
            category='education',
            priority=3,
            description="提供患者用药教育",
            rationale="提高用药依从性和安全性",
            expected_outcome="患者了解用药注意事项，提高自我管理能力",
            alternatives=["制作用药清单", "设置用药提醒", "家属参与用药管理"]
        ))
        
        return {
            'recommendations': recommendations,
            'summary': f"共生成 {len(recommendations)} 条临床建议"
        }


class PatientEducatorAgent(BaseAgent):
    """患者教育代理"""
    
    def __init__(self):
        super().__init__(AgentRole.PATIENT_EDUCATOR, "Patient Educator")
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成患者教育材料"""
        self.log_action("生成患者教育材料", {})
        
        medications = context.get('medications', [])
        alerts = context.get('alerts', [])
        
        education_materials = []
        
        # 为每种药物生成教育材料
        for med in medications:
            material = self._generate_medication_education(med)
            education_materials.append(material)
        
        # 生成安全提示
        safety_tips = self._generate_safety_tips(alerts)
        
        # 生成用药时间表
        medication_schedule = self._generate_medication_schedule(medications)
        
        return {
            'education_materials': education_materials,
            'safety_tips': safety_tips,
            'medication_schedule': medication_schedule,
            'summary': f"已为 {len(medications)} 种药物生成教育材料"
        }
    
    def _generate_medication_education(self, medication: Dict[str, Any]) -> Dict[str, Any]:
        """生成药物教育材料"""
        drug_name = medication.get('name', '未知药物')
        
        return {
            'drug_name': drug_name,
            'purpose': medication.get('indication', '请遵医嘱使用'),
            'how_to_take': {
                'dose': f"{medication.get('dose', '')} {medication.get('unit', '')}",
                'frequency': medication.get('frequency', '遵医嘱'),
                'route': medication.get('route', '口服'),
                'timing': '建议固定时间服用',
                'with_food': '请咨询医生或药师'
            },
            'what_to_expect': [
                '按医嘱规律服药',
                '不要自行调整剂量',
                '如有不适及时就医'
            ],
            'warnings': [
                '不要与他人分享处方药',
                '将药物存放在儿童接触不到的地方',
                '注意药品有效期'
            ],
            'when_to_seek_help': [
                '出现严重不良反应',
                '过敏症状（皮疹、呼吸困难等）',
                '药物过量'
            ]
        }
    
    def _generate_safety_tips(self, alerts: List[SafetyAlert]) -> List[str]:
        """生成安全提示"""
        tips = [
            "按时服药，不要漏服或重复服用",
            "使用药盒或用药提醒工具帮助记忆",
            "定期复查相关指标",
            "就医时告知医生所有正在使用的药物（包括非处方药和保健品）"
        ]
        
        # 基于警报添加特定提示
        for alert in alerts:
            if '肾功能' in alert.title:
                tips.append("注意监测尿量和体重变化")
            if '肝功能' in alert.title:
                tips.append("注意观察皮肤和眼睛是否发黄")
            if '出血' in alert.title.lower():
                tips.append("注意观察是否有异常出血或瘀伤")
        
        return tips
    
    def _generate_medication_schedule(
        self,
        medications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """生成用药时间表"""
        schedule = []
        
        time_slots = {
            'morning': '早餐时',
            'noon': '午餐时',
            'evening': '晚餐时',
            'bedtime': '睡前'
        }
        
        for med in medications:
            frequency = med.get('frequency', '')
            
            if 'qd' in frequency.lower() or '每日一次' in frequency:
                schedule.append({
                    'drug': med.get('name', ''),
                    'time': 'morning',
                    'description': f"{time_slots['morning']}服用 {med.get('dose', '')} {med.get('unit', '')}"
                })
            elif 'bid' in frequency.lower() or '每日两次' in frequency:
                schedule.append({
                    'drug': med.get('name', ''),
                    'time': 'morning',
                    'description': f"{time_slots['morning']}服用 {med.get('dose', '')} {med.get('unit', '')}"
                })
                schedule.append({
                    'drug': med.get('name', ''),
                    'time': 'evening',
                    'description': f"{time_slots['evening']}服用 {med.get('dose', '')} {med.get('unit', '')}"
                })
            elif 'tid' in frequency.lower() or '每日三次' in frequency:
                for slot in ['morning', 'noon', 'evening']:
                    schedule.append({
                        'drug': med.get('name', ''),
                        'time': slot,
                        'description': f"{time_slots[slot]}服用 {med.get('dose', '')} {med.get('unit', '')}"
                    })
        
        return schedule


class SafeRxAgent:
    """SafeRx主代理 - 智能用药安全系统"""
    
    def __init__(
        self,
        ddi_model=None,
        kg_query=None,
        risk_scorer=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        
        # 创建协调者
        self.coordinator = CoordinatorAgent()
        
        # 创建并注册子代理
        self.ddi_analyst = DDIAnalystAgent(ddi_model=ddi_model, kg_query=kg_query)
        self.risk_assessor = RiskAssessorAgent(risk_scorer=risk_scorer)
        self.clinical_advisor = ClinicalAdvisorAgent()
        self.patient_educator = PatientEducatorAgent()
        
        self.coordinator.register_agent(self.ddi_analyst)
        self.coordinator.register_agent(self.risk_assessor)
        self.coordinator.register_agent(self.clinical_advisor)
        self.coordinator.register_agent(self.patient_educator)
        
        logger.info("SafeRx Agent 初始化完成")
    
    async def analyze_prescription(
        self,
        patient_info: Dict[str, Any],
        medications: List[Dict[str, Any]],
        clinical_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """分析处方安全性
        
        Args:
            patient_info: 患者信息
            medications: 药物列表
            clinical_context: 临床上下文
            
        Returns:
            综合分析结果
        """
        context = {
            'patient_info': patient_info,
            'medications': medications,
            'clinical_context': clinical_context or {}
        }
        
        result = await self.coordinator.process(context)
        
        return result
    
    def analyze_prescription_sync(
        self,
        patient_info: Dict[str, Any],
        medications: List[Dict[str, Any]],
        clinical_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """同步版本的处方分析"""
        return asyncio.run(self.analyze_prescription(
            patient_info, medications, clinical_context
        ))
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取代理状态"""
        return {
            'coordinator': {
                'sub_agents': list(self.coordinator.sub_agents.keys()),
                'history_length': len(self.coordinator.conversation_history)
            },
            'ddi_analyst': {
                'history_length': len(self.ddi_analyst.conversation_history)
            },
            'risk_assessor': {
                'history_length': len(self.risk_assessor.conversation_history)
            }
        }


# 使用示例
if __name__ == "__main__":
    async def main():
        # 创建SafeRx Agent
        agent = SafeRxAgent()
        
        # 测试数据
        patient_info = {
            'age': 72,
            'weight': 68.5,
            'egfr': 55.0,
            'alt': 45,
            'ast': 38,
            'comorbidities': ['hypertension', 'diabetes', 'heart_failure'],
            'allergies': ['penicillin']
        }
        
        medications = [
            {
                'name': 'warfarin',
                'dose': 5,
                'unit': 'mg',
                'route': '口服',
                'frequency': 'qd',
                'indication': '房颤抗凝'
            },
            {
                'name': 'aspirin',
                'dose': 100,
                'unit': 'mg',
                'route': '口服',
                'frequency': 'qd',
                'indication': '抗血小板'
            },
            {
                'name': 'metformin',
                'dose': 500,
                'unit': 'mg',
                'route': '口服',
                'frequency': 'bid',
                'indication': '2型糖尿病'
            },
            {
                'name': 'lisinopril',
                'dose': 10,
                'unit': 'mg',
                'route': '口服',
                'frequency': 'qd',
                'indication': '高血压'
            }
        ]
        
        # 分析处方
        result = await agent.analyze_prescription(patient_info, medications)
        
        print("=== SafeRx 处方分析结果 ===")
        print(f"总结: {result['summary']}")
        print(f"\n警报数量: {len(result['alerts'])}")
        for alert in result['alerts']:
            print(f"  [{alert['level']}] {alert['title']}")
        
        print(f"\n建议数量: {len(result['recommendations'])}")
        for rec in result['recommendations']:
            print(f"  [优先级{rec['priority']}] {rec['description']}")
        
        print(f"\n行动项:")
        for item in result['action_items']:
            print(f"  [{item['priority']}] {item['action']} - {item['deadline']}")
    
    asyncio.run(main())