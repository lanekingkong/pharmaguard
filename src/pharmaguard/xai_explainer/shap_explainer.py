"""
SHAP可解释性模块
为DDI预测模型提供特征重要性和决策解释
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """解释结果"""
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    decision_factors: List[Dict[str, Any]]
    counterfactuals: List[Dict[str, Any]]
    visualization_paths: Dict[str, str]
    summary: str


@dataclass
class FeatureInfo:
    """特征信息"""
    name: str
    description: str
    category: str  # molecular, clinical, genetic, etc.
    importance_score: float = 0.0
    shap_value: float = 0.0
    contribution_direction: str = "neutral"  # positive, negative, neutral


class SHAPExplainer:
    """SHAP解释器"""
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        feature_descriptions: Optional[Dict[str, str]] = None,
        background_data: Optional[np.ndarray] = None,
        output_dir: str = "./explanation_results"
    ):
        self.model = model
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions or {}
        self.background_data = background_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化SHAP解释器
        self.explainer = None
        self._initialize_explainer()
        
        # 特征类别映射
        self.feature_categories = self._categorize_features()
        
    def _categorize_features(self) -> Dict[str, str]:
        """特征分类"""
        categories = {}
        
        # 分子特征
        molecular_keywords = ['fingerprint', 'smiles', 'molecular', 'atom', 'bond', 'ring']
        # 靶点特征
        target_keywords = ['target', 'protein', 'enzyme', 'receptor', 'binding']
        # 通路特征
        pathway_keywords = ['pathway', 'metabolic', 'signaling', 'biological_process']
        # 临床特征
        clinical_keywords = ['age', 'weight', 'creatinine', 'egfr', 'alt', 'ast', 'albumin']
        # 基因特征
        genetic_keywords = ['cyp', 'gene', 'genotype', 'variant', 'polymorphism']
        # 患者特征
        patient_keywords = ['comorbidity', 'disease', 'condition', 'allergy']
        # 药物特征
        drug_keywords = ['dose', 'frequency', 'route', 'duration', 'adherence']
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in molecular_keywords):
                categories[feature] = "molecular"
            elif any(keyword in feature_lower for keyword in target_keywords):
                categories[feature] = "target"
            elif any(keyword in feature_lower for keyword in pathway_keywords):
                categories[feature] = "pathway"
            elif any(keyword in feature_lower for keyword in clinical_keywords):
                categories[feature] = "clinical"
            elif any(keyword in feature_lower for keyword in genetic_keywords):
                categories[feature] = "genetic"
            elif any(keyword in feature_lower for keyword in patient_keywords):
                categories[feature] = "patient"
            elif any(keyword in feature_lower for keyword in drug_keywords):
                categories[feature] = "drug"
            else:
                categories[feature] = "other"
        
        return categories
    
    def _initialize_explainer(self):
        """初始化SHAP解释器"""
        if self.background_data is None:
            logger.warning("No background data provided, using dummy data")
            # 创建虚拟背景数据
            self.background_data = np.random.randn(100, len(self.feature_names))
        
        # 转换为PyTorch张量
        background_tensor = torch.FloatTensor(self.background_data)
        
        # 创建SHAP解释器
        self.explainer = shap.DeepExplainer(
            self.model,
            background_tensor
        )
    
    def explain_prediction(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        prediction_threshold: float = 0.5,
        top_k_features: int = 10
    ) -> ExplanationResult:
        """解释单个预测"""
        # 确保输入是张量
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data
        
        # 获取模型预测
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(input_tensor)
            if isinstance(prediction, tuple):
                prediction = prediction[0]  # 取第一个输出
            prediction_prob = torch.sigmoid(prediction).item()
            prediction_binary = 1 if prediction_prob >= prediction_threshold else 0
        
        # 计算SHAP值
        shap_values = self.explainer.shap_values(input_tensor)
        
        # 处理SHAP值（可能返回列表）
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # 取第一个类别的SHAP值
        
        # 转换为numpy数组
        shap_array = shap_values.numpy() if isinstance(shap_values, torch.Tensor) else shap_values
        
        # 计算特征重要性
        feature_importance = self._calculate_feature_importance(shap_array)
        
        # 获取决策因素
        decision_factors = self._extract_decision_factors(
            shap_array, input_tensor.numpy(), top_k_features
        )
        
        # 生成反事实解释
        counterfactuals = self._generate_counterfactuals(
            input_tensor.numpy(), shap_array, prediction_prob
        )
        
        # 创建可视化
        visualization_paths = self._create_visualizations(
            shap_array, feature_importance, input_tensor.numpy()
        )
        
        # 生成总结
        summary = self._generate_summary(
            prediction_prob, prediction_binary, decision_factors, feature_importance
        )
        
        return ExplanationResult(
            prediction=prediction_prob,
            confidence=abs(prediction_prob - 0.5) * 2,  # 距离0.5的归一化距离
            feature_importance=feature_importance,
            decision_factors=decision_factors,
            counterfactuals=counterfactuals,
            visualization_paths=visualization_paths,
            summary=summary
        )
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """计算特征重要性"""
        # 计算绝对SHAP值的平均值
        abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 归一化到0-1
        if abs_shap.sum() > 0:
            importance_scores = abs_shap / abs_shap.sum()
        else:
            importance_scores = np.zeros_like(abs_shap)
        
        # 创建特征重要性字典
        feature_importance = {}
        for i, (feature_name, score) in enumerate(zip(self.feature_names, importance_scores)):
            feature_importance[feature_name] = float(score)
        
        return feature_importance
    
    def _extract_decision_factors(
        self,
        shap_values: np.ndarray,
        input_values: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """提取决策因素"""
        # 获取每个特征的SHAP值
        shap_vals = shap_values.flatten()
        
        # 按绝对值排序
        sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
        
        decision_factors = []
        for idx in sorted_indices[:top_k]:
            feature_idx = idx % len(self.feature_names)
            feature_name = self.feature_names[feature_idx]
            shap_val = shap_vals[idx]
            input_val = input_values.flatten()[idx]
            
            # 确定贡献方向
            if shap_val > 0:
                direction = "positive"
                effect = "增加DDI风险"
            elif shap_val < 0:
                direction = "negative"
                effect = "降低DDI风险"
            else:
                direction = "neutral"
                effect = "无显著影响"
            
            # 获取特征描述
            description = self.feature_descriptions.get(
                feature_name,
                f"特征 {feature_name}"
            )
            
            # 获取特征类别
            category = self.feature_categories.get(feature_name, "unknown")
            
            decision_factors.append({
                'feature_name': feature_name,
                'feature_description': description,
                'category': category,
                'shap_value': float(shap_val),
                'input_value': float(input_val),
                'direction': direction,
                'effect': effect,
                'importance_rank': len(decision_factors) + 1
            })
        
        return decision_factors
    
    def _generate_counterfactuals(
        self,
        input_values: np.ndarray,
        shap_values: np.ndarray,
        prediction_prob: float
    ) -> List[Dict[str, Any]]:
        """生成反事实解释"""
        counterfactuals = []
        
        # 找到最重要的特征
        abs_shap = np.abs(shap_values).flatten()
        top_indices = np.argsort(abs_shap)[::-1][:3]  # 前3个最重要特征
        
        for idx in top_indices:
            feature_idx = idx % len(self.feature_names)
            feature_name = self.feature_names[feature_idx]
            shap_val = shap_values.flatten()[idx]
            current_val = input_values.flatten()[idx]
            
            # 生成反事实值
            if shap_val > 0:  # 正向贡献，降低该值可能降低风险
                cf_value = current_val * 0.5  # 降低50%
                effect = "降低DDI风险"
            else:  # 负向贡献，增加该值可能降低风险
                cf_value = current_val * 1.5  # 增加50%
                effect = "可能进一步降低DDI风险"
            
            counterfactuals.append({
                'feature_name': feature_name,
                'current_value': float(current_val),
                'counterfactual_value': float(cf_value),
                'suggested_change': f"将 {feature_name} 从 {current_val:.2f} 调整为 {cf_value:.2f}",
                'expected_effect': effect,
                'confidence': 'medium'
            })
        
        return counterfactuals
    
    def _create_visualizations(
        self,
        shap_values: np.ndarray,
        feature_importance: Dict[str, float],
        input_values: np.ndarray
    ) -> Dict[str, str]:
        """创建可视化图表"""
        visualization_paths = {}
        
        try:
            # 1. 特征重要性条形图
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            
            # 按类别分组
            categories = {}
            for feature, importance in feature_importance.items():
                category = self.feature_categories.get(feature, "other")
                if category not in categories:
                    categories[category] = []
                categories[category].append((feature, importance))
            
            # 绘制分组条形图
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            y_pos = 0
            for (category, color) in zip(categories.keys(), colors):
                features_in_category = categories[category]
                features_in_category.sort(key=lambda x: x[1], reverse=True)
                
                feature_names = [f[0] for f in features_in_category[:5]]  # 每个类别最多5个
                importances = [f[1] for f in features_in_category[:5]]
                
                ax1.barh(
                    range(y_pos, y_pos + len(feature_names)),
                    importances,
                    color=color,
                    label=category
                )
                
                # 添加特征标签
                for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                    ax1.text(
                        importance + 0.001,
                        y_pos + i,
                        f"{feature} ({importance:.3f})",
                        va='center',
                        fontsize=9
                    )
                
                y_pos += len(feature_names) + 1
            
            ax1.set_xlabel('特征重要性')
            ax1.set_title('DDI预测特征重要性（按类别分组）')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            fig1.tight_layout()
            importance_path = self.output_dir / "feature_importance.png"
            fig1.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            visualization_paths['feature_importance'] = str(importance_path)
            
            # 2. SHAP摘要图
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # 准备数据
            shap_array = shap_values.flatten()
            feature_indices = np.arange(len(self.feature_names))
            
            # 创建散点图
            scatter = ax2.scatter(
                shap_array,
                np.tile(feature_indices, shap_values.shape[0]),
                c=input_values.flatten(),
                cmap='coolwarm',
                alpha=0.6,
                s=50
            )
            
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('SHAP值（对预测的影响）')
            ax2.set_ylabel('特征索引')
            ax2.set_title('SHAP值摘要图')
            ax2.set_yticks(feature_indices)
            ax2.set_yticklabels(self.feature_names)
            plt.colorbar(scatter, ax=ax2, label='特征值')
            ax2.grid(True, alpha=0.3)
            
            fig2.tight_layout()
            summary_path = self.output_dir / "shap_summary.png"
            fig2.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            visualization_paths['shap_summary'] = str(summary_path)
            
            # 3. 决策瀑布图
            fig3, ax3 = plt.subplots(figsize=(14, 8))
            
            # 获取最重要的特征
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            
            feature_names_top = [f[0] for f in top_features]
            shap_vals_top = [shap_values[0, i] for i, f in enumerate(self.feature_names) 
                           if f in feature_names_top]
            
            # 计算累积值
            base_value = 0.5  # 假设基础值
            cumulative = base_value
            values = [base_value]
            
            for shap_val in shap_vals_top:
                cumulative += shap_val
                values.append(cumulative)
            
            # 绘制瀑布图
            ax3.bar(
                range(len(values)),
                values,
                color=['lightgray'] + ['steelblue' if v > 0 else 'coral' for v in shap_vals_top],
                edgecolor='black'
            )
            
            # 添加标签
            ax3.set_xticks(range(len(values)))
            ax3.set_xticklabels(['基础值'] + feature_names_top, rotation=45, ha='right')
            ax3.set_ylabel('预测值')
            ax3.set_title('决策瀑布图（从基础值到最终预测）')
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='决策阈值')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            fig3.tight_layout()
            waterfall_path = self.output_dir / "decision_waterfall.png"
            fig3.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            visualization_paths['decision_waterfall'] = str(waterfall_path)
            
            # 4. 热力图（特征相关性）
            fig4, ax4 = plt.subplots(figsize=(12, 10))
            
            # 计算特征相关性（简化）
            correlation_matrix = np.corrcoef(input_values.T)
            
            sns.heatmap(
                correlation_matrix,
                ax=ax4,
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': '相关系数'},
                xticklabels=self.feature_names,
                yticklabels=self.feature_names
            )
            
            ax4.set_title('特征相关性热力图')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax4.get_yticklabels(), rotation=0)
            
            fig4.tight_layout()
            heatmap_path = self.output_dir / "feature_correlation.png"
            fig4.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            visualization_paths['feature_correlation'] = str(heatmap_path)
            
        except Exception as e:
            logger.error(f"创建可视化时出错: {e}")
        
        return visualization_paths
    
    def _generate_summary(
        self,
        prediction_prob: float,
        prediction_binary: int,
        decision_factors: List[Dict[str, Any]],
        feature_importance: Dict[str, float]
    ) -> str:
        """生成解释总结"""
        # 确定预测结果
        if prediction_binary == 1:
            result_text = f"预测存在DDI（概率: {prediction_prob:.2%}）"
        else:
            result_text = f"预测不存在DDI（概率: {prediction_prob:.2%}）"
        
        # 获取最重要的特征
        top_factors = sorted(decision_factors, key=lambda x: abs(x['shap_value']), reverse=True)[:3]
        
        # 构建总结
        summary_parts = [result_text]
        summary_parts.append("\n主要决策因素：")
        
        for i, factor in enumerate(top_factors, 1):
            direction_symbol = "↑" if factor['direction'] == 'positive' else "↓"
            summary_parts.append(
                f"{i}. {factor['feature_description']} ({direction_symbol}): "
                f"贡献值 {factor['shap_value']:.4f}，{factor['effect']}"
            )
        
        # 添加特征类别信息
        category_importance = {}
        for factor in decision_factors:
            category = factor['category']
            importance = abs(factor['shap_value'])
            category_importance[category] = category_importance.get(category, 0) + importance
        
        if category_importance:
            top_category = max(category_importance.items(), key=lambda x: x[1])
            summary_parts.append(f"\n最重要的特征类别: {top_category[0]}（贡献度: {top_category[1]:.3f}）")
        
        # 添加建议
        summary_parts.append("\n临床建议：")
        if prediction_binary == 1:
            summary_parts.append("1. 考虑调整药物组合以避免相互作用")
            summary_parts.append("2. 加强用药监测，特别是高风险特征")
            summary_parts.append("3. 考虑使用替代药物")
        else:
            summary_parts.append("1. 当前药物组合风险较低")
            summary_parts.append("2. 继续常规用药监测")
            summary_parts.append("3. 如有新症状出现，及时评估")
        
        return "\n".join(summary_parts)
    
    def batch_explain(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        output_file: Optional[str] = None
    ) -> List[ExplanationResult]:
        """批量解释预测"""
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data
        
        explanations = []
        
        for i in range(input_tensor.shape[0]):
            logger.info(f"解释第 {i+1}/{input_tensor.shape[0]} 个样本")
            
            single_input = input_tensor[i:i+1]
            explanation = self.explain_prediction(single_input)
            explanations.append(explanation)
        
        # 保存到文件
        if output_file:
            self._save_explanations(explanations, output_file)
        
        return explanations
    
    def _save_explanations(self, explanations: List[ExplanationResult], output_file: str):
        """保存解释结果"""
        output_path = self.output_dir / output_file
        
        # 转换为可序列化的字典
        serializable_explanations = []
        for exp in explanations:
            exp_dict = {
                'prediction': exp.prediction,
                'confidence': exp.confidence,
                'feature_importance': exp.feature_importance,
                'decision_factors': exp.decision_factors,
                'counterfactuals': exp.counterfactuals,
                'visualization_paths': exp.visualization_paths,
                'summary': exp.summary
            }
            serializable_explanations.append(exp_dict)
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_explanations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"解释结果已保存到: {output_path}")
    
    def generate_report(
        self,
        explanations: List[ExplanationResult],
        report_title: str = "DDI预测可解释性报告"
    ) -> str:
        """生成详细报告"""
        report_parts = [
            f"# {report_title}\n",
            f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析样本数: {len(explanations)}\n",
            "## 1. 总体统计\n"
        ]
        
        # 总体统计
        predictions = [exp.prediction for exp in explanations]
        confidences = [exp.confidence for exp in explanations]
        
        report_parts.extend([
            f"- 平均预测概率: {np.mean(predictions):.3f}",
            f"- 预测标准差: {np.std(predictions):.3f}",
            f"- 平均置信度: {np.mean(confidences):.3f}",
            f"- 高风险预测比例: {sum(1 for p in predictions if p >= 0.7) / len(predictions):.1%}",
            f"- 低风险预测比例: {sum(1 for p in predictions if p <= 0.3) / len(predictions):.1%}\n"
        ])
        
        # 特征重要性汇总
        report_parts.append("## 2. 特征重要性汇总\n")
        
        # 汇总所有样本的特征重要性
        all_importances = {}
        for exp in explanations:
            for feature, importance in exp.feature_importance.items():
                all_importances[feature] = all_importances.get(feature, 0) + importance
        
        # 归一化
        total_importance = sum(all_importances.values())
        if total_importance > 0:
            for feature in all_importances:
                all_importances[feature] /= total_importance
        
        # 按重要性排序
        sorted_importances = sorted(
            all_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # 前10个
        
        report_parts.append("| 排名 | 特征名称 | 平均重要性 | 类别 |")
        report_parts.append("|------|----------|------------|------|")
        
        for i, (feature, importance) in enumerate(sorted_importances, 1):
            category = self.feature_categories.get(feature, "unknown")
            report_parts.append(f"| {i} | {feature} | {importance:.4f} | {category} |")
        
        report_parts.append("\n")
        
        # 决策模式分析
        report_parts.append("## 3. 决策模式分析\n")
        
        # 分析最常见的决策因素
        decision_patterns = {}
        for exp in explanations:
            top_factor = exp.decision_factors[0] if exp.decision_factors else None
            if top_factor:
                pattern_key = f"{top_factor['feature_name']}_{top_factor['direction']}"
                decision_patterns[pattern_key] = decision_patterns.get(pattern_key, 0) + 1
        
        if decision_patterns:
            report_parts.append("最常见的决策模式：")
            for pattern, count in sorted(decision_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                feature, direction = pattern.split('_')
                direction_text = "增加风险" if direction == "positive" else "降低风险"
                percentage = count / len(explanations)
                report_parts.append(f"- {feature} {direction_text}: {percentage:.1%} 的样本")
        
        report_parts.append("\n")
        
        # 临床建议汇总
        report_parts.append("## 4. 临床建议汇总\n")
        
        high_risk_count = sum(1 for exp in explanations if exp.prediction >= 0.7)
        moderate_risk_count = sum(1 for exp in explanations if 0.3 <= exp.prediction < 0.7)
        low_risk_count = sum(1 for exp in explanations if exp.prediction < 0.3)
        
        report_parts.extend([
            f"- 高风险样本: {high_risk_count} 个 ({high_risk_count/len(explanations):.1%})",
            f"- 中风险样本: {moderate_risk_count} 个 ({moderate_risk_count/len(explanations):.1%})",
            f"- 低风险样本: {low_risk_count} 个 ({low_risk_count/len(explanations):.1%})\n"
        ])
        
        report_parts.append("### 高风险样本建议：")
        report_parts.append("1. 立即进行用药重整评估")
        report_parts.append("2. 考虑调整药物组合")
        report_parts.append("3. 加强临床监测和患者教育")
        report_parts.append("4. 制定个体化用药方案\n")
        
        report_parts.append("### 中风险样本建议：")
        report_parts.append("1. 定期评估用药方案")
        report_parts.append("2. 监测相关临床指标")
        report_parts.append("3. 提供用药指导\n")
        
        report_parts.append("### 低风险样本建议：")
        report_parts.append("1. 继续当前治疗方案")
        report_parts.append("2. 常规用药随访")
        report_parts.append("3. 教育患者识别不良反应\n")
        
        # 可视化文件列表
        if explanations and explanations[0].visualization_paths:
            report_parts.append("## 5. 生成的可视化文件\n")
            for vis_name, vis_path in explanations[0].visualization_paths.items():
                report_parts.append(f"- {vis_name}: `{vis_path}`")
        
        return "\n".join(report_parts)


# 使用示例
if __name__ == "__main__":
    # 创建虚拟模型
    class DummyModel(nn.Module):
        def __init__(self, input_dim=50):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # 配置
    input_dim = 50
    feature_names = [f"feature_{i}" for i in range(input_dim)]
    
    # 添加一些有意义的特征名称
    meaningful_features = [
        "drug_a_fingerprint_similarity",
        "drug_b_fingerprint_similarity", 
        "shared_target_count",
        "metabolic_pathway_overlap",
        "patient_age",
        "renal_function_egfr",
        "hepatic_function_alt",
        "cyp2c9_genotype",
        "cyp2d6_genotype",
        "comorbidity_count"
    ]
    
    for i, feat in enumerate(meaningful_features):
        if i < len(feature_names):
            feature_names[i] = feat
    
    feature_descriptions = {
        "drug_a_fingerprint_similarity": "药物A分子指纹相似度",
        "drug_b_fingerprint_similarity": "药物B分子指纹相似度",
        "shared_target_count": "共享靶点数量",
        "metabolic_pathway_overlap": "代谢通路重叠度",
        "patient_age": "患者年龄",
        "renal_function_egfr": "肾功能估算肾小球滤过率",
        "hepatic_function_alt": "肝功能丙氨酸氨基转移酶",
        "cyp2c9_genotype": "CYP2C9基因型",
        "cyp2d6_genotype": "CYP2D6基因型",
        "comorbidity_count": "合并症数量"
    }
    
    # 创建模型和解释器
    model = DummyModel(input_dim=input_dim)
    explainer = SHAPExplainer(
        model=model,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        output_dir="./test_explanations"
    )
    
    # 创建测试数据
    test_data = np.random.randn(5, input_dim)
    
    # 解释单个预测
    print("=== 单个预测解释示例 ===")
    explanation = explainer.explain_prediction(test_data[0:1])
    print(explanation.summary)
    
    # 批量解释
    print("\n=== 批量解释示例 ===")
    explanations = explainer.batch_explain(test_data, "batch_explanations.json")
    print(f"完成 {len(explanations)} 个样本的解释")
    
    # 生成报告
    print("\n=== 生成报告 ===")
    report = explainer.generate_report(explanations)
    
    # 保存报告
    report_path = Path("./test_explanations") / "explanation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {report_path}")
    print(f"可视化文件保存在: {explainer.output_dir}")