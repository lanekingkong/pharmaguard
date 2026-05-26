"""
Polypharmacy Risk Scoring System
多重用药风险评估系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(Enum):
    """风险因素枚举"""
    # 药物相关
    NUM_MEDICATIONS = "number_of_medications"
    DDI_SEVERITY = "ddi_severity"
    NARROW_THERAPEUTIC_INDEX = "narrow_therapeutic_index"
    HIGH_RISK_MEDICATION = "high_risk_medication"
    
    # 患者相关
    AGE = "age"
    RENAL_IMPAIRMENT = "renal_impairment"
    HEPATIC_IMPAIRMENT = "hepatic_impairment"
    GENETIC_VARIANT = "genetic_variant"
    COMORBIDITIES = "comorbidities"
    FRAILTY = "frailty"
    
    # 治疗相关
    DURATION = "treatment_duration"
    ADHERENCE = "adherence_risk"
    MONITORING = "monitoring_requirements"


@dataclass
class Medication:
    """药物信息"""
    name: str
    atc_code: Optional[str] = None
    dose: Optional[float] = None
    unit: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    indications: List[str] = field(default_factory=list)
    is_high_risk: bool = False
    is_nti: bool = False  # 治疗窗狭窄药物
    requires_monitoring: bool = False


@dataclass
class PatientInfo:
    """患者信息"""
    age: int
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    
    # 器官功能
    creatinine: Optional[float] = None  # μmol/L
    egfr: Optional[float] = None  # mL/min/1.73m²
    alt: Optional[float] = None  # U/L
    ast: Optional[float] = None  # U/L
    albumin: Optional[float] = None  # g/L
    
    # 基因型
    cyp2c9: Optional[str] = None
    cyp2c19: Optional[str] = None
    cyp2d6: Optional[str] = None
    cyp3a4: Optional[str] = None
    vkorc1: Optional[str] = None
    
    # 合并症
    comorbidities: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    
    # 其他
    smoking: Optional[bool] = False
    alcohol: Optional[bool] = False
    pregnancy: Optional[bool] = False
    breastfeeding: Optional[bool] = False


@dataclass
class DDIPrediction:
    """DDI预测结果"""
    drug_a: str
    drug_b: str
    has_interaction: bool
    severity: str  # mild, moderate, severe
    confidence: float
    mechanism: Optional[str] = None
    clinical_implications: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class PolypharmacyScorer:
    """多重用药风险评估器"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or self._default_config()
        self._initialize_risk_weights()
        self._initialize_nti_drugs()
        self._initialize_high_risk_drugs()
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'age_threshold': 65,
            'renal_threshold': 60,  # eGFR阈值
            'hepatic_threshold': {
                'alt': 40,  # U/L
                'ast': 40   # U/L
            },
            'max_medications_low': 4,
            'max_medications_moderate': 7,
            'max_medications_high': 10,
            'weights': {
                'num_medications': 0.15,
                'ddi_severity': 0.25,
                'age': 0.10,
                'renal_function': 0.15,
                'hepatic_function': 0.10,
                'genetics': 0.10,
                'comorbidities': 0.10,
                'high_risk_meds': 0.05
            }
        }
    
    def _initialize_risk_weights(self):
        """初始化风险权重"""
        self.risk_weights = self.config['weights']
        
    def _initialize_nti_drugs(self):
        """初始化治疗窗狭窄药物列表"""
        self.nti_drugs = {
            'warfarin', 'digoxin', 'lithium', 'phenytoin', 'theophylline',
            'cyclosporine', 'tacrolimus', 'sirolimus', 'carbamazepine',
            'valproic_acid', 'levothyroxine', 'insulin', 'aminoglycosides'
        }
    
    def _initialize_high_risk_drugs(self):
        """初始化高风险药物列表"""
        self.high_risk_drugs = {
            'anticoagulants': {'warfarin', 'dabigatran', 'rivaroxaban', 'apixaban'},
            'opioids': {'morphine', 'oxycodone', 'fentanyl', 'hydromorphone'},
            'benzodiazepines': {'diazepam', 'lorazepam', 'alprazolam', 'clonazepam'},
            'antipsychotics': {'haloperidol', 'risperidone', 'olanzapine', 'quetiapine'},
            'antidiabetics': {'insulin', 'glibenclamide', 'glimepiride'},
            'chemotherapy': set()  # 化疗药物通常高风险
        }
    
    def calculate_risk_score(
        self,
        medications: List[Medication],
        patient_info: PatientInfo,
        ddi_predictions: List[DDIPrediction]
    ) -> Dict[str, Any]:
        """计算综合风险评分
        
        Args:
            medications: 药物列表
            patient_info: 患者信息
            ddi_predictions: DDI预测结果列表
            
        Returns:
            风险评估结果字典
        """
        logger.info(f"开始风险评估，药物数量: {len(medications)}")
        
        # 计算各项风险分数
        risk_factors = self._calculate_risk_factors(
            medications, patient_info, ddi_predictions
        )
        
        # 计算加权总分
        total_score = self._calculate_weighted_score(risk_factors)
        
        # 确定风险等级
        risk_level = self._determine_risk_level(total_score, risk_factors)
        
        # 生成风险解释
        explanations = self._generate_explanations(risk_factors, total_score, risk_level)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            risk_factors, risk_level, medications, patient_info
        )
        
        result = {
            'overall_risk_score': round(total_score, 2),
            'risk_level': risk_level.value,
            'risk_factors': risk_factors,
            'explanations': explanations,
            'recommendations': recommendations,
            'breakdown': self._get_score_breakdown(risk_factors)
        }
        
        logger.info(f"风险评估完成: 总分={total_score:.2f}, 等级={risk_level.value}")
        return result
    
    def _calculate_risk_factors(
        self,
        medications: List[Medication],
        patient_info: PatientInfo,
        ddi_predictions: List[DDIPrediction]
    ) -> Dict[str, float]:
        """计算各项风险因素分数"""
        risk_factors = {}
        
        # 1. 药物数量风险
        risk_factors['num_medications'] = self._calculate_medication_count_risk(
            len(medications)
        )
        
        # 2. DDI风险
        risk_factors['ddi_severity'] = self._calculate_ddi_risk(ddi_predictions)
        
        # 3. 患者年龄风险
        risk_factors['age'] = self._calculate_age_risk(patient_info.age)
        
        # 4. 肾功能风险
        risk_factors['renal_function'] = self._calculate_renal_risk(
            patient_info.egfr, patient_info.creatinine
        )
        
        # 5. 肝功能风险
        risk_factors['hepatic_function'] = self._calculate_hepatic_risk(
            patient_info.alt, patient_info.ast, patient_info.albumin
        )
        
        # 6. 基因风险
        risk_factors['genetics'] = self._calculate_genetic_risk(
            patient_info, medications
        )
        
        # 7. 合并症风险
        risk_factors['comorbidities'] = self._calculate_comorbidity_risk(
            patient_info.comorbidities
        )
        
        # 8. 高风险药物
        risk_factors['high_risk_meds'] = self._calculate_high_risk_medication_risk(
            medications
        )
        
        # 9. 治疗窗狭窄药物
        risk_factors['nti_drugs'] = self._calculate_nti_drug_risk(medications)
        
        # 10. 药物依从性风险（简化）
        risk_factors['adherence'] = self._estimate_adherence_risk(medications)
        
        return risk_factors
    
    def _calculate_medication_count_risk(self, num_meds: int) -> float:
        """计算药物数量风险"""
        if num_meds <= self.config['max_medications_low']:
            return 0.0
        elif num_meds <= self.config['max_medications_moderate']:
            return 0.3 * (num_meds - self.config['max_medications_low']) / (
                self.config['max_medications_moderate'] - self.config['max_medications_low']
            )
        elif num_meds <= self.config['max_medications_high']:
            return 0.3 + 0.4 * (num_meds - self.config['max_medications_moderate']) / (
                self.config['max_medications_high'] - self.config['max_medications_moderate']
            )
        else:
            return 0.7 + 0.3 * min(1.0, (num_meds - self.config['max_medications_high']) / 10)
    
    def _calculate_ddi_risk(self, ddi_predictions: List[DDIPrediction]) -> float:
        """计算DDI风险"""
        if not ddi_predictions:
            return 0.0
        
        severe_count = sum(1 for ddi in ddi_predictions if ddi.severity == 'severe')
        moderate_count = sum(1 for ddi in ddi_predictions if ddi.severity == 'moderate')
        
        # 加权计算
        risk_score = (
            severe_count * 1.0 +
            moderate_count * 0.5 +
            len([ddi for ddi in ddi_predictions if ddi.severity == 'mild']) * 0.2
        )
        
        # 归一化到0-1
        return min(1.0, risk_score / 5.0)
    
    def _calculate_age_risk(self, age: int) -> float:
        """计算年龄风险"""
        if age < 65:
            return 0.0
        elif age < 75:
            return 0.3
        elif age < 85:
            return 0.6
        else:
            return 0.9
    
    def _calculate_renal_risk(self, egfr: Optional[float], creatinine: Optional[float]) -> float:
        """计算肾功能风险"""
        if egfr is None and creatinine is None:
            return 0.2  # 未知状态，中等风险
        
        if egfr is not None:
            if egfr >= 90:
                return 0.0
            elif egfr >= 60:
                return 0.2
            elif egfr >= 30:
                return 0.5
            elif egfr >= 15:
                return 0.8
            else:
                return 1.0
        
        # 基于肌酐估算（简化）
        if creatinine is not None:
            if creatinine < 100:
                return 0.0
            elif creatinine < 200:
                return 0.3
            elif creatinine < 300:
                return 0.6
            else:
                return 0.9
        
        return 0.0
    
    def _calculate_hepatic_risk(
        self,
        alt: Optional[float],
        ast: Optional[float],
        albumin: Optional[float]
    ) -> float:
        """计算肝功能风险"""
        risk_factors = 0
        
        if alt is not None and alt > self.config['hepatic_threshold']['alt']:
            risk_factors += 1
        
        if ast is not None and ast > self.config['hepatic_threshold']['ast']:
            risk_factors += 1
        
        if albumin is not None and albumin < 35:  # 低白蛋白
            risk_factors += 1
        
        if risk_factors == 0:
            return 0.0
        elif risk_factors == 1:
            return 0.3
        elif risk_factors == 2:
            return 0.6
        else:
            return 0.9
    
    def _calculate_genetic_risk(
        self,
        patient_info: PatientInfo,
        medications: List[Medication]
    ) -> float:
        """计算基因风险"""
        # 简化实现：检查CYP450代谢相关基因型
        risk_score = 0.0
        
        # CYP2C9相关药物（如华法林）
        cyp2c9_drugs = {'warfarin', 'phenytoin', 'losartan'}
        patient_drugs = {med.name.lower() for med in medications}
        
        if any(drug in patient_drugs for drug in cyp2c9_drugs):
            if patient_info.cyp2c9 in ['*2/*2', '*2/*3', '*3/*3']:
                risk_score += 0.5  # 慢代谢型
        
        # CYP2D6相关药物
        cyp2d6_drugs = {'codeine', 'tramadol', 'tamoxifen', 'metoprolol'}
        if any(drug in patient_drugs for drug in cyp2d6_drugs):
            if patient_info.cyp2d6 in ['*4/*4', '*5/*5']:  # 慢代谢型
                risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _calculate_comorbidity_risk(self, comorbidities: List[str]) -> float:
        """计算合并症风险"""
        high_risk_comorbidities = {
            'heart_failure', 'liver_cirrhosis', 'chronic_kidney_disease',
            'dementia', 'parkinson', 'copd', 'diabetes'
        }
        
        risk_count = sum(1 for comorbidity in comorbidities 
                        if comorbidity.lower() in high_risk_comorbidities)
        
        if risk_count == 0:
            return 0.0
        elif risk_count == 1:
            return 0.3
        elif risk_count == 2:
            return 0.6
        else:
            return 0.9
    
    def _calculate_high_risk_medication_risk(self, medications: List[Medication]) -> float:
        """计算高风险药物风险"""
        high_risk_count = sum(1 for med in medications if med.is_high_risk)
        
        if high_risk_count == 0:
            return 0.0
        elif high_risk_count == 1:
            return 0.3
        elif high_risk_count == 2:
            return 0.6
        else:
            return 0.9
    
    def _calculate_nti_drug_risk(self, medications: List[Medication]) -> float:
        """计算治疗窗狭窄药物风险"""
        nti_count = sum(1 for med in medications if med.is_nti)
        
        if nti_count == 0:
            return 0.0
        elif nti_count == 1:
            return 0.4
        elif nti_count == 2:
            return 0.7
        else:
            return 1.0
    
    def _estimate_adherence_risk(self, medications: List[Medication]) -> float:
        """估算依从性风险（简化）"""
        # 基于药物数量、频率、剂型等估算
        num_meds = len(medications)
        
        if num_meds <= 3:
            return 0.1
        elif num_meds <= 5:
            return 0.3
        elif num_meds <= 8:
            return 0.5
        elif num_meds <= 12:
            return 0.7
        else:
            return 0.9
    
    def _calculate_weighted_score(self, risk_factors: Dict[str, float]) -> float:
        """计算加权总分"""
        total_score = 0.0
        total_weight = 0.0
        
        for factor, score in risk_factors.items():
            weight = self.risk_weights.get(factor, 0.05)  # 默认权重
            total_score += score * weight
            total_weight += weight
        
        # 归一化到0-100
        if total_weight > 0:
            normalized_score = (total_score / total_weight) * 100
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    def _determine_risk_level(
        self,
        total_score: float,
        risk_factors: Dict[str, float]
    ) -> RiskLevel:
        """确定风险等级"""
        # 如果有严重DDI或极高风险因素，直接定为高风险
        if risk_factors.get('ddi_severity', 0) > 0.8:
            return RiskLevel.CRITICAL
        
        if total_score < 30:
            return RiskLevel.LOW
        elif total_score < 60:
            return RiskLevel.MODERATE
        elif total_score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_explanations(
        self,
        risk_factors: Dict[str, float],
        total_score: float,
        risk_level: RiskLevel
    ) -> Dict[str, str]:
        """生成风险解释"""
        explanations = {}
        
        # 总体解释
        explanations['overall'] = (
            f"患者多重用药风险评分为{total_score:.1f}/100，"
            f"属于{self._get_chinese_risk_level(risk_level)}风险级别。"
        )
        
        # 关键风险因素
        top_factors = sorted(
            risk_factors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_factors:
            factor_descriptions = []
            for factor, score in top_factors:
                if score > 0.3:  # 只报告显著风险因素
                    desc = self._get_factor_description(factor, score)
                    factor_descriptions.append(desc)
            
            if factor_descriptions:
                explanations['key_factors'] = "主要风险因素包括：" + "；".join(factor_descriptions)
        
        return explanations
    
    def _get_chinese_risk_level(self, risk_level: RiskLevel) -> str:
        """获取中文风险等级描述"""
        mapping = {
            RiskLevel.LOW: "低",
            RiskLevel.MODERATE: "中",
            RiskLevel.HIGH: "高",
            RiskLevel.CRITICAL: "极高"
        }
        return mapping.get(risk_level, "未知")
    
    def _get_factor_description(self, factor: str, score: float) -> str:
        """获取风险因素描述"""
        descriptions = {
            'num_medications': f"用药数量过多（风险评分{score:.1%}）",
            'ddi_severity': f"存在严重的药物相互作用（风险评分{score:.1%}）",
            'age': f"患者年龄较大（风险评分{score:.1%}）",
            'renal_function': f"肾功能受损（风险评分{score:.1%}）",
            'hepatic_function': f"肝功能异常（风险评分{score:.1%}）",
            'genetics': f"存在药物代谢相关基因变异（风险评分{score:.1%}）",
            'comorbidities': f"合并多种基础疾病（风险评分{score:.1%}）",
            'high_risk_meds': f"使用高风险药物（风险评分{score:.1%}）",
            'nti_drugs': f"使用治疗窗狭窄药物（风险评分{score:.1%}）",
            'adherence': f"用药依从性可能较差（风险评分{score:.1%}）"
        }
        return descriptions.get(factor, f"{factor}（风险评分{score:.1%}）")
    
    def _generate_recommendations(
        self,
        risk_factors: Dict[str, float],
        risk_level: RiskLevel,
        medications: List[Medication],
        patient_info: PatientInfo
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于风险等级的建议
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("建议进行用药重整，由临床药师或医生评估")
            recommendations.append("加强用药监测，特别是高风险药物")
            recommendations.append("考虑简化用药方案，减少不必要的药物")
        
        # 基于具体风险因素的建议
        if risk_factors.get('ddi_severity', 0) > 0.5:
            recommendations.append("调整存在严重相互作用的药物组合")
        
        if risk_factors.get('renal_function', 0) > 0.5:
            recommendations.append("根据肾功能调整药物剂量")
            recommendations.append("避免使用肾毒性药物")
        
        if risk_factors.get('num_medications', 0) > 0.5:
            recommendations.append("评估每种药物的必要性和有效性")
            recommendations.append("考虑停用非必需药物")
        
        if risk_factors.get('nti_drugs', 0) > 0.3:
            recommendations.append("加强治疗窗狭窄药物的血药浓度监测")
        
        # 通用建议
        recommendations.append("提供详细的用药教育，提高依从性")
        recommendations.append("定期评估用药方案，至少每3-6个月一次")
        
        return recommendations
    
    def _get_score_breakdown(self, risk_factors: Dict[str, float]) -> Dict[str, float]:
        """获取分数分解"""
        breakdown = {}
        for factor, score in risk_factors.items():
            weight = self.risk_weights.get(factor, 0.05)
            contribution = score * weight * 100  # 转换为0-100的贡献度
            breakdown[factor] = round(contribution, 2)
        return breakdown


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    medications = [
        Medication(name="warfarin", is_high_risk=True, is_nti=True),
        Medication(name="aspirin"),
        Medication(name="metformin"),
        Medication(name="lisinopril"),
        Medication(name="atorvastatin"),
        Medication(name="metoprolol"),
        Medication(name="furosemide"),
    ]
    
    patient_info = PatientInfo(
        age=72,
        weight=68.5,
        egfr=55.0,
        comorbidities=["hypertension", "diabetes", "heart_failure"]
    )
    
    ddi_predictions = [
        DDIPrediction(
            drug_a="warfarin",
            drug_b="aspirin",
            has_interaction=True,
            severity="severe",
            confidence=0.92,
            mechanism="Increased bleeding risk"
        )
    ]
    
    # 计算风险评分
    scorer = PolypharmacyScorer()
    result = scorer.calculate_risk_score(medications, patient_info, ddi_predictions)
    
    print("风险评估结果:")
    print(f"总分: {result['overall_risk_score']}")
    print(f"风险等级: {result['risk_level']}")
    print(f"解释: {result['explanations']['overall']}")
    print(f"关键因素: {result['explanations'].get('key_factors', '无')}")
    print("\n建议:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec}")