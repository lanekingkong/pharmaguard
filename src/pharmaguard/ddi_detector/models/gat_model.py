"""
Graph Attention Network (GAT) for DDI Prediction
基于图注意力网络的药物-药物相互作用预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch


class DrugEncoder(nn.Module):
    """药物特征编码器"""
    
    def __init__(
        self,
        fingerprint_dim: int = 1024,
        target_dim: int = 256,
        pathway_dim: int = 128,
        side_effect_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int = 256
    ):
        super().__init__()
        
        # 分子指纹编码器
        self.fp_encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 靶点编码器
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 通路编码器
        self.pathway_encoder = nn.Sequential(
            nn.Linear(pathway_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 副作用编码器
        self.se_encoder = nn.Sequential(
            nn.Linear(side_effect_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, fingerprint, target, pathway, side_effect):
        """前向传播
        
        Args:
            fingerprint: 分子指纹 [batch_size, fingerprint_dim]
            target: 靶点向量 [batch_size, target_dim]
            pathway: 通路向量 [batch_size, pathway_dim]
            side_effect: 副作用向量 [batch_size, side_effect_dim]
            
        Returns:
            药物嵌入向量 [batch_size, output_dim]
        """
        fp_emb = self.fp_encoder(fingerprint)
        target_emb = self.target_encoder(target)
        pathway_emb = self.pathway_encoder(pathway)
        se_emb = self.se_encoder(side_effect)
        
        # 特征拼接
        combined = torch.cat([fp_emb, target_emb, pathway_emb, se_emb], dim=-1)
        drug_embedding = self.fusion(combined)
        
        return drug_embedding


class DDIGAT(nn.Module):
    """DDI预测的图注意力网络"""
    
    def __init__(
        self,
        drug_embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.drug_embedding_dim = drug_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # GAT层
        self.gat_convs = nn.ModuleList()
        in_channels = drug_embedding_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim // num_heads if i < num_layers - 1 else hidden_dim
            self.gat_convs.append(
                GATConv(
                    in_channels,
                    out_channels,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
            in_channels = hidden_dim
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 患者特征维度
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 256),  # 药物A + 药物B + 上下文
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # 严重程度回归器
        self.severity_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1之间的严重程度分数
        )
        
        # 注意力权重提取（用于可解释性）
        self.attention_weights = None
        
    def forward(self, drug_a_emb, drug_b_emb, context, edge_index, batch=None):
        """前向传播
        
        Args:
            drug_a_emb: 药物A嵌入 [num_drugs, drug_embedding_dim]
            drug_b_emb: 药物B嵌入 [num_drugs, drug_embedding_dim]
            context: 患者上下文特征 [batch_size, context_dim]
            edge_index: 图边索引 [2, num_edges]
            batch: 批次索引 [num_drugs]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
            severity: 严重程度分数 [batch_size, 1]
            attention_weights: 注意力权重列表
        """
        # 构建药物对图
        x = torch.cat([drug_a_emb, drug_b_emb], dim=0)  # 拼接所有药物节点
        num_drugs = drug_a_emb.size(0)
        
        # 通过GAT层
        self.attention_weights = []
        for conv in self.gat_convs:
            x, attention = conv(x, edge_index, return_attention_weights=True)
            self.attention_weights.append(attention)
            if conv != self.gat_convs[-1]:
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # 分离药物A和药物B的节点表示
        drug_a_nodes = x[:num_drugs]
        drug_b_nodes = x[num_drugs:]
        
        # 全局池化（如果提供批次信息）
        if batch is not None:
            drug_a_pooled = global_mean_pool(drug_a_nodes, batch[:num_drugs])
            drug_b_pooled = global_mean_pool(drug_b_nodes, batch[num_drugs:])
        else:
            drug_a_pooled = drug_a_nodes.mean(dim=0, keepdim=True)
            drug_b_pooled = drug_b_nodes.mean(dim=0, keepdim=True)
        
        # 编码上下文
        context_emb = self.context_encoder(context)
        
        # 特征拼接
        combined_features = torch.cat([
            drug_a_pooled,
            drug_b_pooled,
            context_emb
        ], dim=-1)
        
        # 分类和严重程度预测
        logits = self.classifier(combined_features)
        severity = self.severity_regressor(combined_features)
        
        return logits, severity
    
    def predict_proba(self, drug_a_emb, drug_b_emb, context, edge_index, batch=None):
        """预测概率
        
        Returns:
            probabilities: 各类别概率 [batch_size, num_classes]
            severity: 严重程度分数 [batch_size, 1]
        """
        self.eval()
        with torch.no_grad():
            logits, severity = self.forward(drug_a_emb, drug_b_emb, context, edge_index, batch)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities, severity
    
    def get_attention_maps(self):
        """获取注意力权重图（用于可解释性）"""
        return self.attention_weights


class MultiModalDDIModel(nn.Module):
    """多模态DDI预测模型（集成GAT和Transformer）"""
    
    def __init__(
        self,
        fingerprint_dim: int = 1024,
        target_dim: int = 256,
        pathway_dim: int = 128,
        side_effect_dim: int = 64,
        context_dim: int = 10,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        transformer_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        # 药物编码器
        self.drug_encoder = DrugEncoder(
            fingerprint_dim=fingerprint_dim,
            target_dim=target_dim,
            pathway_dim=pathway_dim,
            side_effect_dim=side_effect_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # GAT模型
        self.gat_model = DDIGAT(
            drug_embedding_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes
        )
        
        # Transformer编码器（用于序列建模）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # GAT + Transformer + 原始特征
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 最终分类器
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 严重程度预测器
        self.final_severity = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, drug_a_features, drug_b_features, context, edge_index, batch=None):
        """前向传播
        
        Args:
            drug_a_features: 药物A特征元组 (fingerprint, target, pathway, side_effect)
            drug_b_features: 药物B特征元组
            context: 患者上下文特征
            edge_index: 图边索引
            batch: 批次索引
            
        Returns:
            logits: 分类logits
            severity: 严重程度分数
            attention_weights: 注意力权重
        """
        # 编码药物特征
        drug_a_emb = self.drug_encoder(*drug_a_features)
        drug_b_emb = self.drug_encoder(*drug_b_features)
        
        # GAT预测
        gat_logits, gat_severity = self.gat_model(
            drug_a_emb, drug_b_emb, context, edge_index, batch
        )
        
        # Transformer编码
        drug_features = torch.stack([drug_a_emb, drug_b_emb], dim=1)  # [batch, 2, hidden_dim]
        transformer_out = self.transformer_encoder(drug_features)
        transformer_pooled = transformer_out.mean(dim=1)  # 平均池化
        
        # 特征融合
        gat_features = torch.cat([
            drug_a_emb.mean(dim=0, keepdim=True) if batch is None else 
            global_mean_pool(drug_a_emb, batch[:drug_a_emb.size(0)]),
            drug_b_emb.mean(dim=0, keepdim=True) if batch is None else 
            global_mean_pool(drug_b_emb, batch[drug_a_emb.size(0):])
        ], dim=-1)
        
        fused_features = self.fusion_layer(
            torch.cat([gat_features, transformer_pooled, drug_a_emb + drug_b_emb], dim=-1)
        )
        
        # 结合上下文
        context_enhanced = torch.cat([fused_features, context], dim=-1)
        
        # 最终预测
        final_logits = self.final_classifier(context_enhanced)
        final_severity = self.final_severity(context_enhanced)
        
        return final_logits, final_severity, self.gat_model.get_attention_maps()
    
    def predict(self, drug_a_features, drug_b_features, context, edge_index, batch=None):
        """预测方法
        
        Returns:
            prediction: 预测类别
            probability: 预测概率
            severity: 严重程度
            attention: 注意力权重
        """
        self.eval()
        with torch.no_grad():
            logits, severity, attention = self.forward(
                drug_a_features, drug_b_features, context, edge_index, batch
            )
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
        return {
            'prediction': predictions.cpu().numpy(),
            'probability': probabilities.cpu().numpy(),
            'severity': severity.cpu().numpy(),
            'attention_weights': attention
        }


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    fingerprint_dim = 1024
    target_dim = 256
    pathway_dim = 128
    side_effect_dim = 64
    context_dim = 10
    
    # 创建测试数据
    drug_a_fp = torch.randn(batch_size, fingerprint_dim)
    drug_a_target = torch.randn(batch_size, target_dim)
    drug_a_pathway = torch.randn(batch_size, pathway_dim)
    drug_a_se = torch.randn(batch_size, side_effect_dim)
    
    drug_b_fp = torch.randn(batch_size, fingerprint_dim)
    drug_b_target = torch.randn(batch_size, target_dim)
    drug_b_pathway = torch.randn(batch_size, pathway_dim)
    drug_b_se = torch.randn(batch_size, side_effect_dim)
    
    context = torch.randn(batch_size, context_dim)
    
    # 创建图结构（完全连接）
    num_nodes = batch_size * 2
    edge_index = []
    for i in range(batch_size):
        for j in range(batch_size):
            edge_index.append([i, j + batch_size])
            edge_index.append([j + batch_size, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 测试模型
    model = MultiModalDDIModel(
        fingerprint_dim=fingerprint_dim,
        target_dim=target_dim,
        pathway_dim=pathway_dim,
        side_effect_dim=side_effect_dim,
        context_dim=context_dim
    )
    
    drug_a_features = (drug_a_fp, drug_a_target, drug_a_pathway, drug_a_se)
    drug_b_features = (drug_b_fp, drug_b_target, drug_b_pathway, drug_b_se)
    
    output = model(drug_a_features, drug_b_features, context, edge_index)
    print(f"Logits shape: {output[0].shape}")
    print(f"Severity shape: {output[1].shape}")
    print(f"Number of attention layers: {len(output[2])}")