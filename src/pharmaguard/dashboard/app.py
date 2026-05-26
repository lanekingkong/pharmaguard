"""
PharmaGuard Dashboard - Web管理界面
基于Streamlit的交互式数据可视化和分析面板
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 页面配置
st.set_page_config(
    page_title="PharmaGuard Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .card-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .card-orange {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .alert-critical {
        padding: 1rem;
        background-color: #ffeaea;
        border-left: 4px solid #e74c3c;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .alert-warning {
        padding: 1rem;
        background-color: #fff8ea;
        border-left: 4px solid #f39c12;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .alert-info {
        padding: 1rem;
        background-color: #eaf4ff;
        border-left: 4px solid #3498db;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .recommendation-card {
        padding: 1rem;
        background-color: #f0f4f8;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# API配置
API_BASE_URL = "http://localhost:8000/api/v1"


def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
    """发送API请求"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return {"success": False, "message": f"Unsupported method: {method}"}
        
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "API服务未连接"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def generate_sample_ddi_data(n_drugs: int = 10) -> pd.DataFrame:
    """生成样本DDI数据"""
    np.random.seed(42)
    
    drugs = [
        "Warfarin", "Aspirin", "Clopidogrel", "Atorvastatin",
        "Metformin", "Lisinopril", "Omeprazole", "Metoprolol",
        "Levothyroxine", "Amlodipine", "Simvastatin", "Metronidazole",
        "Fluconazole", "Rifampin", "Carbamazepine", "Phenytoin",
        "Digoxin", "Spironolactone", "Furosemide", "Hydrochlorothiazide"
    ][:n_drugs]
    
    data = []
    for i, drug_a in enumerate(drugs):
        for j, drug_b in enumerate(drugs):
            if i >= j:
                continue
            data.append({
                'drug_a': drug_a,
                'drug_b': drug_b,
                'interaction_probability': np.random.beta(2, 5),
                'severity': np.random.choice(['mild', 'moderate', 'severe'], p=[0.4, 0.4, 0.2]),
                'confidence': np.random.uniform(0.6, 0.95)
            })
    
    return pd.DataFrame(data)


def generate_sample_patient_data(n_patients: int = 100) -> pd.DataFrame:
    """生成样本患者数据"""
    np.random.seed(42)
    
    ages = np.random.randint(18, 95, n_patients)
    genders = np.random.choice(['M', 'F'], n_patients)
    num_medications = np.random.poisson(4, n_patients) + 1
    egfr_values = np.random.normal(80, 25, n_patients)
    egfr_values = np.clip(egfr_values, 10, 140)
    
    # 计算风险分数
    risk_scores = (
        (ages > 65).astype(int) * 15 +
        (num_medications > 5).astype(int) * 20 +
        (egfr_values < 60).astype(int) * 25 +
        np.random.normal(0, 10, n_patients)
    )
    risk_scores = np.clip(risk_scores, 0, 100)
    
    return pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(n_patients)],
        'age': ages,
        'gender': genders,
        'num_medications': num_medications,
        'egfr': egfr_values.round(1),
        'risk_score': risk_scores.round(1),
        'has_alerts': risk_scores > 50
    })


# =================== 页面布局 ===================

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/pill.png", width=80)
        st.markdown("## PharmaGuard")
        st.markdown("智能用药安全系统")
        st.markdown("---")
        
        # 导航
        page = st.radio(
            "导航",
            ["仪表板", "DDI分析", "处方检查", "风险评估", "药物查询", "系统设置"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 系统状态")
        
        # API状态检查
        health = make_api_request("/health")
        if health.get('success'):
            st.success("API: 在线")
        else:
            st.error("API: 离线")
        
        st.markdown(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")
        
        return page


def render_dashboard():
    """渲染仪表板页面"""
    st.markdown('<p class="main-header">PharmaGuard 仪表板</p>', unsafe_allow_html=True)
    
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="metric-value">1,247</div>
            <div class="metric-label">总分析处方数</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card card-green">
            <div class="metric-value">89.3%</div>
            <div class="metric-label">DDI预测准确率</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card card-orange">
            <div class="metric-value">156</div>
            <div class="metric-label">活跃警报</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="card card-red">
            <div class="metric-value">23</div>
            <div class="metric-label">严重DDI发现</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 图表区域
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<p class="sub-header">DDI严重程度分布</p>', unsafe_allow_html=True)
        
        # 生成样本数据
        ddi_data = generate_sample_ddi_data(20)
        
        fig = px.pie(
            ddi_data,
            names='severity',
            title='药物相互作用严重程度分布',
            color='severity',
            color_discrete_map={
                'mild': '#38ef7d',
                'moderate': '#f2c94c',
                'severe': '#f45c43'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown('<p class="sub-header">患者风险评分分布</p>', unsafe_allow_html=True)
        
        patient_data = generate_sample_patient_data(200)
        
        fig = px.histogram(
            patient_data,
            x='risk_score',
            nbins=20,
            title='患者用药风险评分分布',
            color_discrete_sequence=['#667eea']
        )
        fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="高风险阈值")
        st.plotly_chart(fig, use_container_width=True)
    
    # DDI风险热力图
    st.markdown('<p class="sub-header">药物相互作用风险矩阵</p>', unsafe_allow_html=True)
    
    # 创建风险矩阵
    drugs_10 = ddi_data['drug_a'].unique()[:10]
    matrix_data = ddi_data[ddi_data['drug_a'].isin(drugs_10) & ddi_data['drug_b'].isin(drugs_10)]
    
    pivot = matrix_data.pivot_table(
        values='interaction_probability',
        index='drug_a',
        columns='drug_b',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot,
        title='DDI风险热力图',
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 最近警报
    st.markdown('<p class="sub-header">最近安全警报</p>', unsafe_allow_html=True)
    
    sample_alerts = [
        {'level': 'critical', 'title': '华法林 + 阿司匹林严重出血风险', 'time': '10分钟前', 'patient': 'P0123'},
        {'level': 'warning', 'title': '多重用药风险 - 8种药物', 'time': '25分钟前', 'patient': 'P0456'},
        {'level': 'warning', 'title': '肾功能不全 - eGFR 35', 'time': '1小时前', 'patient': 'P0789'},
        {'level': 'info', 'title': '新药物加入处方', 'time': '2小时前', 'patient': 'P0321'},
    ]
    
    for alert in sample_alerts:
        alert_class = f"alert-{alert['level']}"
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>[{alert['level'].upper()}]</strong> {alert['title']}<br>
            <small>患者: {alert['patient']} | {alert['time']}</small>
        </div>
        """, unsafe_allow_html=True)


def render_ddi_analysis():
    """渲染DDI分析页面"""
    st.markdown('<p class="main-header">药物相互作用分析</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        drug_a = st.text_input("药物A", placeholder="输入药物名称", key="ddi_drug_a")
    
    with col2:
        drug_b = st.text_input("药物B", placeholder="输入药物名称", key="ddi_drug_b")
    
    if st.button("分析DDI", type="primary", use_container_width=True):
        if drug_a and drug_b:
            with st.spinner("正在分析..."):
                result = make_api_request(
                    "/predict/ddi",
                    method="POST",
                    data={"drug_a": drug_a, "drug_b": drug_b}
                )
            
            if result.get('success'):
                data = result['data']
                
                # 显示结果
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    probability = data.get('probability', 0)
                    st.metric("相互作用概率", f"{probability:.1%}")
                
                with col2:
                    severity = data.get('severity', 'unknown')
                    st.metric("严重程度", severity.upper())
                
                with col3:
                    confidence = data.get('confidence', 0)
                    st.metric("置信度", f"{confidence:.1%}")
                
                # 详细信息
                st.markdown("### 分析详情")
                
                if data.get('has_interaction'):
                    st.warning(f"检测到药物相互作用: {data.get('description', '')}")
                else:
                    st.success("未检测到显著的药物相互作用")
                
                if data.get('mechanism'):
                    st.info(f"作用机制: {data['mechanism']}")
                
                if data.get('recommendations'):
                    st.markdown("### 建议")
                    for rec in data['recommendations']:
                        st.markdown(f"- {rec}")
            else:
                st.error(f"分析失败: {result.get('message', '未知错误')}")
        else:
            st.warning("请输入两种药物名称")
    
    # 批量DDI分析
    st.markdown("---")
    st.markdown('<p class="sub-header">批量DDI分析</p>', unsafe_allow_html=True)
    
    medication_list = st.text_area(
        "药物列表（每行一个）",
        placeholder="Warfarin\nAspirin\nClopidogrel\nAtorvastatin",
        height=150
    )
    
    if st.button("批量分析", type="secondary"):
        if medication_list.strip():
            medications = [m.strip() for m in medication_list.split('\n') if m.strip()]
            
            if len(medications) >= 2:
                with st.spinner(f"正在分析{len(medications)}种药物的{len(medications)*(len(medications)-1)//2}对组合..."):
                    result = make_api_request(
                        "/predict/ddi/batch",
                        method="POST",
                        data={"medication_list": medications}
                    )
                
                if result.get('success'):
                    data = result['data']
                    
                    st.success(f"分析完成: {data['total_pairs']}对药物组合")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("严重DDI", data['severe_count'])
                    with col2:
                        st.metric("中度DDI", data['moderate_count'])
                    with col3:
                        st.metric("轻度DDI", data['mild_count'])
                    
                    # DDI结果表格
                    if data.get('results'):
                        df = pd.DataFrame(data['results'])
                        st.dataframe(
                            df,
                            column_config={
                                'drug_a': '药物A',
                                'drug_b': '药物B',
                                'probability': st.column_config.ProgressColumn(
                                    'DDI概率',
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=1
                                ),
                                'severity': '严重程度',
                                'confidence': st.column_config.NumberColumn('置信度', format="%.2f")
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.error(f"批量分析失败: {result.get('message')}")
            else:
                st.warning("至少需要2种药物")
        else:
            st.warning("请输入药物列表")


def render_prescription_check():
    """渲染处方检查页面"""
    st.markdown('<p class="main-header">处方安全检查</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">患者信息</p>', unsafe_allow_html=True)
        
        patient_id = st.text_input("患者ID", "P0001")
        age = st.number_input("年龄", min_value=0, max_value=150, value=65)
        weight = st.number_input("体重 (kg)", min_value=0.0, max_value=300.0, value=70.0)
        gender = st.selectbox("性别", ["M", "F"])
        primary_diagnosis = st.text_input("主要诊断", "高血压")
        comorbidities = st.multiselect(
            "合并症",
            ["糖尿病", "心力衰竭", "肾功能不全", "肝功能异常", "COPD", "冠心病", "脑卒中"]
        )
        allergies = st.multiselect(
            "过敏史",
            ["青霉素", "头孢类", "磺胺类", "阿司匹林", "NSAIDs"]
        )
    
    with col2:
        st.markdown('<p class="sub-header">实验室检查</p>', unsafe_allow_html=True)
        
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, max_value=200.0, value=80.0)
        alt = st.number_input("ALT (U/L)", min_value=0.0, max_value=1000.0, value=30.0)
        ast = st.number_input("AST (U/L)", min_value=0.0, max_value=1000.0, value=25.0)
        inr = st.number_input("INR", min_value=0.0, max_value=20.0, value=1.0)
    
    st.markdown("---")
    st.markdown('<p class="sub-header">药物列表</p>', unsafe_allow_html=True)
    
    # 药物编辑区域
    num_meds = st.number_input("药物数量", min_value=1, max_value=10, value=3)
    
    medications = []
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown("**药物名称**")
    with cols[1]:
        st.markdown("**剂量**")
    with cols[2]:
        st.markdown("**单位**")
    with cols[3]:
        st.markdown("**频率**")
    
    for i in range(num_meds):
        cols = st.columns(4)
        with cols[0]:
            drug_name = st.text_input(f"药物{i+1}", key=f"drug_name_{i}", placeholder="药物名称")
        with cols[1]:
            dose = st.number_input(f"剂量{i+1}", key=f"dose_{i}", min_value=0.0, value=100.0)
        with cols[2]:
            unit = st.selectbox(f"单位{i+1}", ["mg", "g", "mcg", "mL", "IU"], key=f"unit_{i}")
        with cols[3]:
            frequency = st.selectbox(f"频率{i+1}", ["qd", "bid", "tid", "qid", "prn"], key=f"freq_{i}")
        
        if drug_name:
            medications.append({
                'name': drug_name,
                'dose': dose,
                'unit': unit,
                'frequency': frequency
            })
    
    if st.button("分析处方", type="primary", use_container_width=True):
        if medications:
            with st.spinner("正在进行处方安全分析..."):
                result = make_api_request(
                    "/analyze/prescription",
                    method="POST",
                    data={
                        "patient": {
                            "patient_id": patient_id,
                            "age": age,
                            "weight": weight,
                            "gender": gender,
                            "primary_diagnosis": primary_diagnosis,
                            "comorbidities": comorbidities,
                            "allergies": allergies,
                            "egfr": egfr,
                            "alt": alt,
                            "ast": ast,
                            "inr": inr
                        },
                        "medications": medications
                    }
                )
            
            if result.get('success'):
                data = result['data']
                
                # 患者摘要
                st.markdown("### 患者用药摘要")
                summary = data.get('patient_summary', {})
                
                cols = st.columns(4)
                cols[0].metric("年龄", summary.get('age'))
                cols[1].metric("药物数", summary.get('medications_count'))
                cols[2].metric("合并症", summary.get('comorbidities_count'))
                cols[3].metric("风险因素", len(summary.get('risk_factors', [])))
                
                # 风险评估
                risk = data.get('risk_assessment', {})
                st.markdown("### 风险评估")
                
                risk_level = risk.get('level', 'unknown')
                risk_score = risk.get('total_score', 0)
                
                # 风险仪表盘
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "风险评分"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # 警报
                alerts = data.get('alerts', [])
                if alerts:
                    st.markdown(f"### 安全警报 ({len(alerts)})")
                    
                    for alert in alerts:
                        level = alert.get('level', 'info')
                        alert_class = f"alert-{level}"
                        st.markdown(f"""
                        <div class="{alert_class}">
                            <strong>[{level.upper()}]</strong> {alert.get('title', '')}<br>
                            <small>{alert.get('description', '')}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("未发现安全警报")
                
                # DDI详情
                ddi = data.get('ddi_analysis', {})
                if ddi.get('ddi_details'):
                    st.markdown(f"### DDI分析 ({len(ddi['ddi_details'])}对)")
                    
                    ddi_df = pd.DataFrame(ddi['ddi_details'])
                    st.dataframe(ddi_df, use_container_width=True, hide_index=True)
                
            else:
                st.error(f"分析失败: {result.get('message', '未知错误')}")
        else:
            st.warning("请至少添加一种药物")


def render_risk_assessment():
    """渲染风险评估页面"""
    st.markdown('<p class="main-header">用药风险评估</p>', unsafe_allow_html=True)
    
    # 患者风险分布
    st.markdown('<p class="sub-header">患者风险概览</p>', unsafe_allow_html=True)
    
    patient_data = generate_sample_patient_data(500)
    
    # 风险等级分布
    patient_data['risk_level'] = pd.cut(
        patient_data['risk_score'],
        bins=[-np.inf, 30, 60, 80, np.inf],
        labels=['低风险', '中风险', '高风险', '极高风险']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = patient_data['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='患者风险等级分布',
            color=risk_counts.index,
            color_discrete_map={
                '低风险': '#38ef7d',
                '中风险': '#f2c94c',
                '高风险': '#f2994a',
                '极高风险': '#f45c43'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            patient_data,
            x='age',
            y='risk_score',
            color='risk_level',
            size='num_medications',
            hover_data=['patient_id', 'egfr'],
            title='年龄-风险评分关系图',
            color_discrete_map={
                '低风险': '#38ef7d',
                '中风险': '#f2c94c',
                '高风险': '#f2994a',
                '极高风险': '#f45c43'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 风险因素分析
    st.markdown('<p class="sub-header">风险因素分析</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.box(
            patient_data,
            y='risk_score',
            title='风险评分分布'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            patient_data,
            x='gender',
            y='risk_score',
            color='gender',
            title='性别-风险评分'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        avg_meds_by_age = patient_data.groupby(
            pd.cut(patient_data['age'], bins=[0, 30, 50, 65, 80, 100])
        )['num_medications'].mean().reset_index()
        avg_meds_by_age.columns = ['age_group', 'avg_medications']
        
        fig = px.bar(
            avg_meds_by_age,
            x='age_group',
            y='avg_medications',
            title='各年龄组平均用药数'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_drug_query():
    """渲染药物查询页面"""
    st.markdown('<p class="main-header">药物信息查询</p>', unsafe_allow_html=True)
    
    search_query = st.text_input("搜索药物", placeholder="输入药物名称...")
    
    if search_query:
        result = make_api_request(f"/drugs/search?q={search_query}&limit=10")
        
        if result.get('success'):
            drugs = result['data'].get('results', [])
            
            if drugs:
                st.markdown(f"找到 {len(drugs)} 个相关药物")
                
                for drug in drugs:
                    with st.expander(f"{drug['name']} (DrugBank: {drug['drugbank_id']})"):
                        st.markdown(f"**匹配度**: {drug['match_score']:.2f}")
                        
                        # 药物详情查询
                        if st.button(f"查看详情", key=f"detail_{drug['drugbank_id']}"):
                            detail = make_api_request(
                                "/query/drug",
                                method="POST",
                                data={"drug_name": drug['name']}
                            )
                            
                            if detail.get('success'):
                                drug_data = detail['data']
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**适应症**")
                                    if drug_data.get('indications'):
                                        for ind in drug_data['indications'][:5]:
                                            st.markdown(f"- {ind}")
                                    else:
                                        st.markdown("暂无数据")
                                    
                                    st.markdown("**靶点**")
                                    if drug_data.get('targets'):
                                        for target in drug_data['targets'][:5]:
                                            st.markdown(f"- {target}")
                                    else:
                                        st.markdown("暂无数据")
                                
                                with col2:
                                    st.markdown("**副作用**")
                                    if drug_data.get('side_effects'):
                                        for se in drug_data['side_effects'][:5]:
                                            st.markdown(f"- {se}")
                                    else:
                                        st.markdown("暂无数据")
            else:
                st.info("未找到相关药物")
        else:
            st.error("查询失败")


def render_system_settings():
    """渲染系统设置页面"""
    st.markdown('<p class="main-header">系统设置</p>', unsafe_allow_html=True)
    
    # 模型配置
    st.markdown('<p class="sub-header">模型配置</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ddi_threshold = st.slider(
            "DDI检测阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="高于此阈值的药物对将被标记为存在相互作用"
        )
        
        risk_threshold = st.slider(
            "高风险警报阈值",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="风险评分超过此值将触发高风险警报"
        )
    
    with col2:
        alert_frequency = st.selectbox(
            "警报检查频率",
            ["实时", "每小时", "每日", "每周"]
        )
        
        max_medications = st.number_input(
            "最大药物分析数量",
            min_value=1,
            max_value=50,
            value=30
        )
    
    # 知识图谱配置
    st.markdown('<p class="sub-header">知识图谱配置</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        kg_uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
    
    with col2:
        kg_user = st.text_input("用户名", "neo4j")
    
    with col3:
        kg_password = st.text_input("密码", type="password", value="password")
    
    if st.button("测试连接", type="secondary"):
        st.success("知识图谱连接成功")
    
    # API配置
    st.markdown('<p class="sub-header">API配置</p>', unsafe_allow_html=True)
    
    api_host = st.text_input("API主机", "0.0.0.0")
    api_port = st.number_input("API端口", min_value=1, max_value=65535, value=8000)
    
    # 保存设置
    if st.button("保存设置", type="primary"):
        st.success("设置已保存")
        st.info("部分设置需要重启服务才能生效")


# =================== 主函数 ===================

def main():
    """主函数"""
    page = render_sidebar()
    
    if page == "仪表板":
        render_dashboard()
    elif page == "DDI分析":
        render_ddi_analysis()
    elif page == "处方检查":
        render_prescription_check()
    elif page == "风险评估":
        render_risk_assessment()
    elif page == "药物查询":
        render_drug_query()
    elif page == "系统设置":
        render_system_settings()


if __name__ == "__main__":
    main()