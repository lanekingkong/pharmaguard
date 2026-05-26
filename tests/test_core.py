"""
PharmaGuard Tests
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestDDIModel:
    """测试DDI模型"""
    
    @pytest.fixture
    def sample_drug_features(self):
        """生成样本药物特征"""
        return {
            'molecular_fingerprint': torch.randn(2048),
            'target_vector': torch.randn(128),
            'pathway_vector': torch.randn(64),
            'side_effect_vector': torch.randn(64),
            'drug_name': 'test_drug'
        }
    
    def test_feature_dimensions(self, sample_drug_features):
        """测试特征维度"""
        assert sample_drug_features['molecular_fingerprint'].shape == (2048,)
        assert sample_drug_features['target_vector'].shape == (128,)
        assert sample_drug_features['pathway_vector'].shape == (64,)
        assert sample_drug_features['side_effect_vector'].shape == (64,)
    
    def test_drug_encoder_creation(self):
        """测试药物编码器创建"""
        try:
            from pharmaguard.ddi_model.gat_model import DrugEncoder
            
            encoder = DrugEncoder(
                fingerprint_dim=2048,
                target_dim=128,
                pathway_dim=64,
                side_effect_dim=64,
                hidden_dim=256,
                output_dim=128
            )
            assert encoder is not None
        except ImportError:
            pytest.skip("DDI model module not available")
    
    def test_forward_pass(self, sample_drug_features):
        """测试前向传播"""
        try:
            from pharmaguard.ddi_model.gat_model import DrugEncoder
            
            encoder = DrugEncoder(
                fingerprint_dim=2048,
                target_dim=128,
                pathway_dim=64,
                side_effect_dim=64,
                hidden_dim=256,
                output_dim=128
            )
            
            output = encoder(
                sample_drug_features['molecular_fingerprint'].unsqueeze(0),
                sample_drug_features['target_vector'].unsqueeze(0),
                sample_drug_features['pathway_vector'].unsqueeze(0),
                sample_drug_features['side_effect_vector'].unsqueeze(0)
            )
            
            assert output.shape == (1, 128)
        except ImportError:
            pytest.skip("DDI model module not available")


class TestRiskScore:
    """测试风险评分"""
    
    def test_polypharmacy_scorer_creation(self):
        """测试风险评分器创建"""
        try:
            from pharmaguard.polypharmacy_scorer import PolypharmacyScorer
            
            scorer = PolypharmacyScorer()
            assert scorer is not None
        except ImportError:
            pytest.skip("Risk scorer module not available")
    
    def test_risk_score_calculation(self):
        """测试风险评分计算"""
        try:
            from pharmaguard.polypharmacy_scorer import PolypharmacyScorer
            
            scorer = PolypharmacyScorer()
            
            medications = [
                {'name': 'warfarin', 'dose': 5.0, 'unit': 'mg'},
                {'name': 'aspirin', 'dose': 100.0, 'unit': 'mg'},
                {'name': 'metformin', 'dose': 500.0, 'unit': 'mg'}
            ]
            
            patient_context = {
                'age': 72,
                'egfr': 55.0,
                'comorbidities': ['hypertension', 'diabetes']
            }
            
            result = scorer.calculate_score(medications, patient_context)
            
            assert 'total_score' in result
            assert 'level' in result
            assert 0 <= result['total_score'] <= 100
        except ImportError:
            pytest.skip("Risk scorer module not available")


class TestKnowledgeGraph:
    """测试知识图谱"""
    
    def test_node_types(self):
        """测试节点类型定义"""
        try:
            from pharmaguard.knowledge_graph import NodeType
            
            assert NodeType.DRUG.value == 'Drug'
            assert NodeType.PROTEIN.value == 'Protein'
            assert NodeType.DISEASE.value == 'Disease'
        except ImportError:
            pytest.skip("Knowledge graph module not available")
    
    def test_relationship_types(self):
        """测试关系类型定义"""
        try:
            from pharmaguard.knowledge_graph import RelationshipType
            
            assert RelationshipType.TARGETS.value == 'TARGETS'
            assert RelationshipType.INDICATES.value == 'INDICATES'
        except ImportError:
            pytest.skip("Knowledge graph module not available")
    
    def test_similarity_calculation(self):
        """测试药物相似度计算"""
        try:
            from pharmaguard.knowledge_graph import MolecularSimilarityCalculator
            
            calc = MolecularSimilarityCalculator()
            
            # 使用模拟SMILES
            smiles_a = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
            smiles_b = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Same
            
            similarity = calc.calculate(smiles_a, smiles_b)
            
            assert 0 <= similarity <= 1
            assert similarity > 0.99  # 相同分子应高度相似
        except ImportError:
            pytest.skip("Knowledge graph module not available")


class TestAPIServer:
    """测试API服务"""
    
    def test_health_endpoint(self):
        """测试健康检查端点"""
        from fastapi.testclient import TestClient
        
        try:
            from pharmaguard.api.api_server import app
            
            client = TestClient(app)
            response = client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert data['data']['status'] == 'healthy'
        except ImportError:
            pytest.skip("API module not available")
    
    def test_ddi_prediction_endpoint(self):
        """测试DDI预测端点"""
        from fastapi.testclient import TestClient
        
        try:
            from pharmaguard.api.api_server import app
            
            client = TestClient(app)
            response = client.post("/api/v1/predict/ddi", json={
                "drug_a": "warfarin",
                "drug_b": "aspirin"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert 'drug_a' in data['data']
            assert 'drug_b' in data['data']
        except ImportError:
            pytest.skip("API module not available")
    
    def test_analyze_prescription_endpoint(self):
        """测试处方分析端点"""
        from fastapi.testclient import TestClient
        
        try:
            from pharmaguard.api.api_server import app
            
            client = TestClient(app)
            response = client.post("/api/v1/analyze/prescription", json={
                "patient": {
                    "age": 72,
                    "egfr": 55.0,
                    "comorbidities": ["hypertension"]
                },
                "medications": [
                    {"name": "warfarin", "dose": 5.0, "unit": "mg"},
                    {"name": "aspirin", "dose": 100.0, "unit": "mg"}
                ]
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
        except ImportError:
            pytest.skip("API module not available")
    
    def test_input_validation(self):
        """测试输入验证"""
        from fastapi.testclient import TestClient
        
        try:
            from pharmaguard.api.api_server import app
            
            client = TestClient(app)
            
            # 测试空药物列表
            response = client.post("/api/v1/analyze/prescription", json={
                "patient": {"age": 30},
                "medications": []
            })
            assert response.status_code == 422
            
            # 测试重复药物
            response = client.post("/api/v1/analyze/prescription", json={
                "patient": {"age": 30},
                "medications": [
                    {"name": "drug_a", "dose": 1.0, "unit": "mg"},
                    {"name": "drug_a", "dose": 1.0, "unit": "mg"}
                ]
            })
            assert response.status_code == 422
        except ImportError:
            pytest.skip("API module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])