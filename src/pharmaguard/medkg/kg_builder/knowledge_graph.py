"""
药物知识图谱构建与查询系统
基于Neo4j的药物-靶点-疾病-通路多关系知识图谱
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import Neo4jError
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型枚举"""
    DRUG = "Drug"
    PROTEIN = "Protein"
    DISEASE = "Disease"
    PATHWAY = "Pathway"
    GENE = "Gene"
    SIDE_EFFECT = "SideEffect"
    PHARMACOLOGICAL_CLASS = "PharmacologicalClass"


class RelationshipType(Enum):
    """关系类型枚举"""
    # 药物相关
    INTERACTS_WITH = "INTERACTS_WITH"
    TARGETS = "TARGETS"
    INDICATES = "INDICATES"  # 适应症
    CAUSES = "CAUSES"  # 引起副作用
    CONTRAINDICATES = "CONTRAINDICATES"  # 禁忌症
    
    # 生物相关
    PARTICIPATES_IN = "PARTICIPATES_IN"  # 蛋白参与通路
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # 基因-疾病关联
    ENCODES = "ENCODES"  # 基因编码蛋白
    REGULATES = "REGULATES"  # 调控关系
    
    # 分类相关
    BELONGS_TO = "BELONGS_TO"  # 属于某分类
    HAS_PARENT = "HAS_PARENT"  # 父子关系


@dataclass
class KGNode:
    """知识图谱节点"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    
    def to_cypher(self) -> str:
        """转换为Cypher创建语句"""
        labels_str = f":{self.node_type.value}"
        if self.labels:
            labels_str += ":" + ":".join(self.labels)
        
        props_str = json.dumps(self.properties, ensure_ascii=False)
        return f"({self.node_id} {labels_str} {props_str})"


@dataclass
class KGRelationship:
    """知识图谱关系"""
    source_id: str
    target_id: str
    rel_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_cypher(self) -> str:
        """转换为Cypher创建语句"""
        props_str = json.dumps(self.properties, ensure_ascii=False)
        return f"({self.source_id})-[:{self.rel_type.value} {props_str}]->({self.target_id})"


@dataclass
class DrugData:
    """药物数据"""
    name: str
    drugbank_id: Optional[str] = None
    atc_codes: List[str] = field(default_factory=list)
    smiles: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    targets: List[Dict[str, Any]] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)
    indications: List[str] = field(default_factory=list)
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    pharmacokinetics: Optional[Dict[str, Any]] = None
    pharmacodynamics: Optional[Dict[str, Any]] = None
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "pharmaguard"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self._initialize_constraints()
        
    def _initialize_constraints(self):
        """初始化约束和索引"""
        constraints_queries = [
            # 唯一性约束
            "CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE",
            "CREATE CONSTRAINT protein_id_unique IF NOT EXISTS FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE",
            "CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.mesh_id IS UNIQUE",
            "CREATE CONSTRAINT pathway_id_unique IF NOT EXISTS FOR (p:Pathway) REQUIRE p.kegg_id IS UNIQUE",
            
            # 索引
            "CREATE INDEX drug_name_index IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX drug_atc_index IF NOT EXISTS FOR (d:Drug) ON (d.atc_codes)",
            "CREATE INDEX protein_name_index IF NOT EXISTS FOR (p:Protein) ON (p.name)",
            "CREATE INDEX disease_name_index IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            
            # 全文搜索索引
            "CREATE FULLTEXT INDEX drug_search IF NOT EXISTS FOR (d:Drug) ON EACH [d.name, d.synonyms]",
            "CREATE FULLTEXT INDEX disease_search IF NOT EXISTS FOR (d:Disease) ON EACH [d.name, d.synonyms]",
        ]
        
        with self.driver.session(database=self.database) as session:
            for query in constraints_queries:
                try:
                    session.run(query)
                except Neo4jError as e:
                    logger.warning(f"Failed to create constraint/index: {e}")
    
    def build_from_drugbank(self, drugbank_data: List[DrugData]):
        """从DrugBank数据构建知识图谱"""
        logger.info(f"开始构建知识图谱，药物数量: {len(drugbank_data)}")
        
        with self.driver.session(database=self.database) as session:
            # 清空现有数据
            session.run("MATCH (n) DETACH DELETE n")
            
            # 批量创建节点和关系
            for i, drug_data in enumerate(drugbank_data, 1):
                if i % 100 == 0:
                    logger.info(f"处理进度: {i}/{len(drugbank_data)}")
                
                self._create_drug_node(session, drug_data)
                
                # 创建靶点节点和关系
                for target in drug_data.targets:
                    self._create_protein_node(session, target)
                    self._create_target_relationship(session, drug_data, target)
                
                # 创建适应症关系
                for indication in drug_data.indications:
                    self._create_disease_node_if_not_exists(session, indication)
                    self._create_indication_relationship(session, drug_data, indication)
                
                # 创建副作用关系
                for side_effect in drug_data.side_effects:
                    self._create_side_effect_node(session, side_effect)
                    self._create_side_effect_relationship(session, drug_data, side_effect)
                
                # 创建DDI关系
                for interaction in drug_data.interactions:
                    self._create_ddi_relationship(session, drug_data, interaction)
            
            # 创建通路和分类关系
            self._create_pathway_and_classification_relationships(session, drugbank_data)
        
        logger.info("知识图谱构建完成")
    
    def _create_drug_node(self, session: Session, drug_data: DrugData):
        """创建药物节点"""
        query = """
        MERGE (d:Drug {drugbank_id: $drugbank_id})
        SET d.name = $name,
            d.atc_codes = $atc_codes,
            d.smiles = $smiles,
            d.molecular_formula = $molecular_formula,
            d.molecular_weight = $molecular_weight,
            d.categories = $categories,
            d.pharmacokinetics = $pharmacokinetics,
            d.pharmacodynamics = $pharmacodynamics
        RETURN d
        """
        
        session.run(
            query,
            drugbank_id=drug_data.drugbank_id or f"DRUG_{hash(drug_data.name)}",
            name=drug_data.name,
            atc_codes=drug_data.atc_codes,
            smiles=drug_data.smiles,
            molecular_formula=drug_data.molecular_formula,
            molecular_weight=drug_data.molecular_weight,
            categories=drug_data.categories,
            pharmacokinetics=drug_data.pharmacokinetics,
            pharmacodynamics=drug_data.pharmacodynamics
        )
    
    def _create_protein_node(self, session: Session, target: Dict[str, Any]):
        """创建蛋白节点"""
        if not target.get('uniprot_id'):
            return
        
        query = """
        MERGE (p:Protein {uniprot_id: $uniprot_id})
        SET p.name = $name,
            p.gene_name = $gene_name,
            p.organism = $organism,
            p.function = $function
        RETURN p
        """
        
        session.run(
            query,
            uniprot_id=target['uniprot_id'],
            name=target.get('name'),
            gene_name=target.get('gene_name'),
            organism=target.get('organism'),
            function=target.get('function')
        )
    
    def _create_target_relationship(self, session: Session, drug_data: DrugData, target: Dict[str, Any]):
        """创建药物-靶点关系"""
        query = """
        MATCH (d:Drug {drugbank_id: $drugbank_id})
        MATCH (p:Protein {uniprot_id: $uniprot_id})
        MERGE (d)-[r:TARGETS]->(p)
        SET r.action = $action,
            r.affinity = $affinity,
            r.mechanism = $mechanism
        RETURN r
        """
        
        session.run(
            query,
            drugbank_id=drug_data.drugbank_id or f"DRUG_{hash(drug_data.name)}",
            uniprot_id=target.get('uniprot_id'),
            action=target.get('action'),
            affinity=target.get('affinity'),
            mechanism=target.get('mechanism')
        )
    
    def _create_disease_node_if_not_exists(self, session: Session, disease_name: str):
        """创建疾病节点（如果不存在）"""
        query = """
        MERGE (d:Disease {name: $name})
        SET d.mesh_id = $mesh_id
        RETURN d
        """
        
        # 简化：使用名称的hash作为mesh_id
        mesh_id = f"MESH_{hash(disease_name) % 1000000}"
        
        session.run(
            query,
            name=disease_name,
            mesh_id=mesh_id
        )
    
    def _create_indication_relationship(self, session: Session, drug_data: DrugData, indication: str):
        """创建适应症关系"""
        query = """
        MATCH (d:Drug {drugbank_id: $drugbank_id})
        MATCH (dis:Disease {name: $indication})
        MERGE (d)-[r:INDICATES]->(dis)
        SET r.approval_status = $approval_status,
            r.evidence_level = $evidence_level
        RETURN r
        """
        
        session.run(
            query,
            drugbank_id=drug_data.drugbank_id or f"DRUG_{hash(drug_data.name)}",
            indication=indication,
            approval_status="approved",  # 简化
            evidence_level="clinical"  # 简化
        )
    
    def _create_side_effect_node(self, session: Session, side_effect: Dict[str, Any]):
        """创建副作用节点"""
        query = """
        MERGE (se:SideEffect {name: $name})
        SET se.frequency = $frequency,
            se.severity = $severity,
            se.organ_system = $organ_system
        RETURN se
        """
        
        session.run(
            query,
            name=side_effect.get('name'),
            frequency=side_effect.get('frequency'),
            severity=side_effect.get('severity'),
            organ_system=side_effect.get('organ_system')
        )
    
    def _create_side_effect_relationship(self, session: Session, drug_data: DrugData, side_effect: Dict[str, Any]):
        """创建副作用关系"""
        query = """
        MATCH (d:Drug {drugbank_id: $drugbank_id})
        MATCH (se:SideEffect {name: $name})
        MERGE (d)-[r:CAUSES]->(se)
        SET r.frequency = $frequency,
            r.severity = $severity,
            r.evidence = $evidence
        RETURN r
        """
        
        session.run(
            query,
            drugbank_id=drug_data.drugbank_id or f"DRUG_{hash(drug_data.name)}",
            name=side_effect.get('name'),
            frequency=side_effect.get('frequency'),
            severity=side_effect.get('severity'),
            evidence=side_effect.get('evidence', 'reported')
        )
    
    def _create_ddi_relationship(self, session: Session, drug_data: DrugData, interaction: Dict[str, Any]):
        """创建药物相互作用关系"""
        query = """
        MATCH (d1:Drug {drugbank_id: $drugbank_id1})
        MATCH (d2:Drug {drugbank_id: $drugbank_id2})
        MERGE (d1)-[r:INTERACTS_WITH]-(d2)
        SET r.description = $description,
            r.severity = $severity,
            r.mechanism = $mechanism,
            r.evidence_level = $evidence_level,
            r.management = $management
        RETURN r
        """
        
        session.run(
            query,
            drugbank_id1=drug_data.drugbank_id or f"DRUG_{hash(drug_data.name)}",
            drugbank_id2=interaction.get('drugbank_id'),
            description=interaction.get('description'),
            severity=interaction.get('severity'),
            mechanism=interaction.get('mechanism'),
            evidence_level=interaction.get('evidence_level'),
            management=interaction.get('management')
        )
    
    def _create_pathway_and_classification_relationships(self, session: Session, drugbank_data: List[DrugData]):
        """创建通路和分类关系"""
        # 简化实现：创建通路节点和分类关系
        pathways = set()
        categories = set()
        
        for drug_data in drugbank_data:
            pathways.update(drug_data.pathways)
            categories.update(drug_data.categories)
        
        # 创建通路节点
        for pathway in pathways:
            query = """
            MERGE (p:Pathway {kegg_id: $kegg_id})
            SET p.name = $name,
                p.description = $description
            RETURN p
            """
            
            session.run(
                query,
                kegg_id=f"PATHWAY_{hash(pathway)}",
                name=pathway,
                description=f"Pathway for {pathway}"
            )
        
        # 创建分类节点
        for category in categories:
            query = """
            MERGE (c:PharmacologicalClass {name: $name})
            SET c.description = $description
            RETURN c
            """
            
            session.run(
                query,
                name=category,
                description=f"Pharmacological class: {category}"
            )
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()


class KnowledgeGraphQuery:
    """知识图谱查询器"""
    
    def __init__(self, driver: Driver, database: str = "pharmaguard"):
        self.driver = driver
        self.database = database
    
    def find_drug_by_name(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """根据名称查找药物"""
        query = """
        MATCH (d:Drug)
        WHERE d.name CONTAINS $name OR $name IN d.synonyms
        RETURN d,
               [(d)-[:TARGETS]->(p:Protein) | p] as targets,
               [(d)-[:INDICATES]->(dis:Disease) | dis] as indications,
               [(d)-[:CAUSES]->(se:SideEffect) | se] as side_effects
        LIMIT 1
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name)
            record = result.single()
            
            if record:
                return self._format_drug_record(record)
        
        return None
    
    def find_ddi_by_drugs(self, drug_a: str, drug_b: str) -> Optional[Dict[str, Any]]:
        """查找两个药物之间的相互作用"""
        query = """
        MATCH (d1:Drug {name: $drug_a})-[r:INTERACTS_WITH]-(d2:Drug {name: $drug_b})
        RETURN r,
               d1,
               d2
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, drug_a=drug_a, drug_b=drug_b)
            record = result.single()
            
            if record:
                return {
                    'drug_a': dict(record['d1']),
                    'drug_b': dict(record['d2']),
                    'interaction': dict(record['r'])
                }
        
        return None
    
    def find_all_ddi_for_drug(self, drug_name: str) -> List[Dict[str, Any]]:
        """查找药物的所有DDI"""
        query = """
        MATCH (d:Drug {name: $name})-[r:INTERACTS_WITH]-(other:Drug)
        RETURN other.name as other_drug,
               r.severity as severity,
               r.mechanism as mechanism,
               r.description as description
        ORDER BY 
          CASE r.severity
            WHEN 'severe' THEN 1
            WHEN 'moderate' THEN 2
            WHEN 'mild' THEN 3
            ELSE 4
          END
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name)
            return [dict(record) for record in result]
    
    def find_drugs_by_target(self, target_name: str) -> List[Dict[str, Any]]:
        """根据靶点查找药物"""
        query = """
        MATCH (d:Drug)-[:TARGETS]->(p:Protein)
        WHERE p.name CONTAINS $name OR p.gene_name CONTAINS $name
        RETURN d.name as drug_name,
               p.name as target_name,
               p.gene_name as gene_name
        ORDER BY drug_name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=target_name)
            return [dict(record) for record in result]
    
    def find_drugs_for_disease(self, disease_name: str) -> List[Dict[str, Any]]:
        """根据疾病查找药物"""
        query = """
        MATCH (d:Drug)-[r:INDICATES]->(dis:Disease)
        WHERE dis.name CONTAINS $name
        RETURN d.name as drug_name,
               dis.name as disease_name,
               r.approval_status as approval_status
        ORDER BY drug_name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=disease_name)
            return [dict(record) for record in result]
    
    def find_common_targets(self, drug_a: str, drug_b: str) -> List[Dict[str, Any]]:
        """查找两个药物的共同靶点"""
        query = """
        MATCH (d1:Drug {name: $drug_a})-[:TARGETS]->(p:Protein)<-[:TARGETS]-(d2:Drug {name: $drug_b})
        RETURN p.name as target_name,
               p.gene_name as gene_name,
               p.function as function
        ORDER BY target_name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, drug_a=drug_a, drug_b=drug_b)
            return [dict(record) for record in result]
    
    def find_drug_similarity(self, drug_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """查找相似药物（基于共同靶点）"""
        query = """
        MATCH (d1:Drug {name: $name})-[:TARGETS]->(p:Protein)<-[:TARGETS]-(d2:Drug)
        WHERE d1 <> d2
        RETURN d2.name as similar_drug,
               COUNT(p) as common_targets,
               COLLECT(p.name) as target_names
        ORDER BY common_targets DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name, limit=limit)
            return [dict(record) for record in result]
    
    def find_drug_pathways(self, drug_name: str) -> List[Dict[str, Any]]:
        """查找药物作用的通路"""
        query = """
        MATCH (d:Drug {name: $name})-[:TARGETS]->(p:Protein)-[:PARTICIPATES_IN]->(path:Pathway)
        RETURN path.name as pathway_name,
               COUNT(DISTINCT p) as target_count,
               COLLECT(DISTINCT p.name) as target_names
        ORDER BY target_count DESC
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name)
            return [dict(record) for record in result]
    
    def find_potential_ddi_by_mechanism(self, drug_name: str) -> List[Dict[str, Any]]:
        """基于机制预测潜在DDI"""
        query = """
        // 查找共享代谢酶的药物
        MATCH (d1:Drug {name: $name})-[:TARGETS]->(p1:Protein)
        WHERE p1.function CONTAINS 'enzyme' AND (p1.name CONTAINS 'CYP' OR p1.name CONTAINS 'UGT')
        MATCH (d2:Drug)-[:TARGETS]->(p2:Protein)
        WHERE p2.name = p1.name AND d1 <> d2
        RETURN d2.name as potential_ddi_drug,
               p1.name as shared_enzyme,
               'metabolic_competition' as mechanism,
               'potential' as evidence_level
        UNION
        // 查找共享转运体的药物
        MATCH (d1:Drug {name: $name})-[:TARGETS]->(p1:Protein)
        WHERE p1.function CONTAINS 'transporter'
        MATCH (d2:Drug)-[:TARGETS]->(p2:Protein)
        WHERE p2.name = p1.name AND d1 <> d2
        RETURN d2.name as potential_ddi_drug,
               p1.name as shared_transporter,
               'transporter_competition' as mechanism,
               'potential' as evidence_level
        ORDER BY potential_ddi_drug
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name)
            return [dict(record) for record in result]
    
    def find_alternative_drugs(self, drug_name: str, disease: str) -> List[Dict[str, Any]]:
        """查找替代药物"""
        query = """
        // 查找治疗相同疾病的不同药物
        MATCH (d1:Drug {name: $name})-[:INDICATES]->(dis:Disease {name: $disease})
        MATCH (d2:Drug)-[:INDICATES]->(dis)
        WHERE d1 <> d2
        OPTIONAL MATCH (d1)-[r:INTERACTS_WITH]-(d2)
        RETURN d2.name as alternative_drug,
               dis.name as disease,
               CASE WHEN r IS NULL THEN 'no_interaction' ELSE r.severity END as interaction_with_original,
               CASE WHEN r IS NULL THEN 'safe' ELSE 'caution' END as recommendation
        ORDER BY recommendation, alternative_drug
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name, disease=disease)
            return [dict(record) for record in result]
    
    def find_drug_contraindications(self, drug_name: str) -> List[Dict[str, Any]]:
        """查找药物禁忌症"""
        query = """
        MATCH (d:Drug {name: $name})-[:CONTRAINDICATES]->(dis:Disease)
        RETURN dis.name as disease_name,
               dis.description as disease_description
        ORDER BY disease_name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=drug_name)
            return [dict(record) for record in result]
    
    def _format_drug_record(self, record) -> Dict[str, Any]:
        """格式化药物记录"""
        drug = dict(record['d'])
        
        # 处理列表属性
        drug['targets'] = [dict(target) for target in record['targets']]
        drug['indications'] = [dict(indication) for indication in record['indications']]
        drug['side_effects'] = [dict(se) for se in record['side_effects']]
        
        return drug
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        queries = {
            'node_counts': """
                MATCH (n)
                RETURN labels(n)[0] as label, COUNT(n) as count
                ORDER BY label
            """,
            'relationship_counts': """
                MATCH ()-[r]->()
                RETURN type(r) as type, COUNT(r) as count
                ORDER BY type
            """,
            'drug_stats': """
                MATCH (d:Drug)
                RETURN 
                    COUNT(d) as total_drugs,
                    COUNT(DISTINCT d.categories) as categories,
                    AVG(SIZE([(d)-[:TARGETS]->() | 1])) as avg_targets_per_drug,
                    MAX(SIZE([(d)-[:TARGETS]->() | 1])) as max_targets
            """,
            'ddi_stats': """
                MATCH ()-[r:INTERACTS_WITH]-()
                WITH r.severity as severity, COUNT(r) as count
                RETURN 
                    severity,
                    count,
                    100.0 * count / SUM(count) OVER() as percentage
                ORDER BY 
                    CASE severity
                        WHEN 'severe' THEN 1
                        WHEN 'moderate' THEN 2
                        WHEN 'mild' THEN 3
                        ELSE 4
                    END
            """
        }
        
        stats = {}
        with self.driver.session(database=self.database) as session:
            for key, query in queries.items():
                result = session.run(query)
                if key == 'node_counts' or key == 'relationship_counts':
                    stats[key] = {record['label']: record['count'] for record in result}
                elif key == 'drug_stats':
                    stats[key] = dict(result.single())
                elif key == 'ddi_stats':
                    stats[key] = [dict(record) for record in result]
        
        return stats


class MolecularSimilarityCalculator:
    """分子相似度计算器"""
    
    def __init__(self, kg_query: KnowledgeGraphQuery):
        self.kg_query = kg_query
    
    def calculate_similarity(self, drug_a: str, drug_b: str) -> Dict[str, Any]:
        """计算两个药物的相似度"""
        # 获取药物信息
        drug_a_info = self.kg_query.find_drug_by_name(drug_a)
        drug_b_info = self.kg_query.find_drug_by_name(drug_b)
        
        if not drug_a_info or not drug_b_info:
            return {'error': 'Drug not found'}
        
        similarities = {}
        
        # 1. 分子结构相似度
        if drug_a_info.get('smiles') and drug_b_info.get('smiles'):
            similarities['structural'] = self._calculate_structural_similarity(
                drug_a_info['smiles'], drug_b_info['smiles']
            )
        
        # 2. 靶点相似度
        targets_a = {t['name'] for t in drug_a_info.get('targets', []) if t.get('name')}
        targets_b = {t['name'] for t in drug_b_info.get('targets', []) if t.get('name')}
        
        if targets_a and targets_b:
            similarities['target'] = self._calculate_jaccard_similarity(targets_a, targets_b)
        
        # 3. 适应症相似度
        indications_a = set(drug_a_info.get('indications', []))
        indications_b = set(drug_b_info.get('indications', []))
        
        if indications_a and indications_b:
            similarities['indication'] = self._calculate_jaccard_similarity(
                indications_a, indications_b
            )
        
        # 4. 副作用相似度
        side_effects_a = {se['name'] for se in drug_a_info.get('side_effects', []) if se.get('name')}
        side_effects_b = {se['name'] for se in drug_b_info.get('side_effects', []) if se.get('name')}
        
        if side_effects_a and side_effects_b:
            similarities['side_effect'] = self._calculate_jaccard_similarity(
                side_effects_a, side_effects_b
            )
        
        # 计算综合相似度
        if similarities:
            similarities['overall'] = sum(similarities.values()) / len(similarities)
        
        return {
            'drug_a': drug_a,
            'drug_b': drug_b,
            'similarities': similarities,
            'common_features': self._find_common_features(drug_a_info, drug_b_info)
        }
    
    def _calculate_structural_similarity(self, smiles_a: str, smiles_b: str) -> float:
        """计算分子结构相似度（基于Tanimoto系数）"""
        try:
            mol_a = Chem.MolFromSmiles(smiles_a)
            mol_b = Chem.MolFromSmiles(smiles_b)
            
            if not mol_a or not mol_b:
                return 0.0
            
            fp_a = AllChem.GetMorganFingerprint(mol_a, 2)
            fp_b = AllChem.GetMorganFingerprint(mol_b, 2)
            
            return AllChem.DataStructs.TanimotoSimilarity(fp_a, fp_b)
        except:
            return 0.0
    
    def _calculate_jaccard_similarity(self, set_a: Set, set_b: Set) -> float:
        """计算Jaccard相似度"""
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_common_features(self, drug_a_info: Dict[str, Any], drug_b_info: Dict[str, Any]) -> Dict[str, List]:
        """查找共同特征"""
        common = {}
        
        # 共同靶点
        targets_a = {t['name'] for t in drug_a_info.get('targets', []) if t.get('name')}
        targets_b = {t['name'] for t in drug_b_info.get('targets', []) if t.get('name')}
        common['targets'] = list(targets_a & targets_b)
        
        # 共同适应症
        indications_a = set(drug_a_info.get('indications', []))
        indications_b = set(drug_b_info.get('indications', []))
        common['indications'] = list(indications_a & indications_b)
        
        # 共同副作用
        side_effects_a = {se['name'] for se in drug_a_info.get('side_effects', []) if se.get('name')}
        side_effects_b = {se['name'] for se in drug_b_info.get('side_effects', []) if se.get('name')}
        common['side_effects'] = list(side_effects_a & side_effects_b)
        
        return common


# 使用示例
if __name__ == "__main__":
    # 创建知识图谱构建器
    builder = KnowledgeGraphBuilder(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    # 创建查询器
    query = KnowledgeGraphQuery(builder.driver)
    
    # 示例查询
    print("=== 药物查询示例 ===")
    drug_info = query.find_drug_by_name("warfarin")
    if drug_info:
        print(f"药物: {drug_info.get('name')}")
        print(f"靶点数量: {len(drug_info.get('targets', []))}")
        print(f"适应症: {', '.join(drug_info.get('indications', []))}")
    
    print("\n=== DDI查询示例 ===")
    ddi_info = query.find_ddi_by_drugs("warfarin", "aspirin")
    if ddi_info:
        print(f"相互作用: {ddi_info['interaction'].get('description')}")
        print(f"严重程度: {ddi_info['interaction'].get('severity')}")
    
    print("\n=== 相似药物查询示例 ===")
    similar_drugs = query.find_drug_similarity("warfarin", limit=5)
    for i, drug in enumerate(similar_drugs, 1):
        print(f"{i}. {drug['similar_drug']} (共同靶点: {drug['common_targets']})")
    
    # 关闭连接
    builder.close()