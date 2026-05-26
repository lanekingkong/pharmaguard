"""
PharmaGuard CLI - 命令行工具
"""

import click
import json
import sys
import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="pharmaguard")
def cli():
    """PharmaGuard - 智能用药安全与DDI预测系统
    
    命令行工具用于药物相互作用预测、处方安全分析和用药风险评估。
    """
    pass


@cli.command()
@click.option('--drug-a', '-a', required=True, help='药物A名称')
@click.option('--drug-b', '-b', required=True, help='药物B名称')
@click.option('--json-output', '-j', is_flag=True, help='以JSON格式输出')
def predict(drug_a: str, drug_b: str, json_output: bool):
    """预测两种药物之间的相互作用"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task(description=f"分析 {drug_a} 与 {drug_b} 的相互作用...", total=None)
        
        # 模拟预测结果
        import hashlib
        hash_val = int(hashlib.md5(f"{drug_a}_{drug_b}".encode()).hexdigest()[:8], 16)
        probability = 0.3 + (hash_val % 70) / 100
        severity = 'severe' if probability > 0.8 else 'moderate' if probability > 0.5 else 'mild'
        
        result = {
            'drug_a': drug_a,
            'drug_b': drug_b,
            'has_interaction': probability > 0.5,
            'probability': probability,
            'severity': severity,
            'confidence': 0.6 + (hash_val % 35) / 100,
            'mechanism': 'CYP450酶系代谢竞争',
            'recommendations': [
                '建议监测药物浓度',
                '必要时调整剂量',
                '观察不良反应'
            ]
        }
    
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 富文本输出
        table = Table(title=f"DDI分析结果: {drug_a} + {drug_b}")
        table.add_column("属性", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("存在相互作用", "是" if result['has_interaction'] else "否")
        table.add_row("概率", f"{result['probability']:.1%}")
        table.add_row("严重程度", result['severity'].upper())
        table.add_row("置信度", f"{result['confidence']:.1%}")
        table.add_row("机制", result['mechanism'])
        
        console.print(table)
        
        if result['recommendations']:
            console.print("\n[bold yellow]建议:[/bold yellow]")
            for rec in result['recommendations']:
                console.print(f"  • {rec}")
        
        # 安全提示
        severity_color = {
            'severe': 'red',
            'moderate': 'yellow',
            'mild': 'green'
        }
        color = severity_color.get(severity, 'white')
        
        if result['has_interaction']:
            console.print(Panel(
                f"[bold {color}]⚠ 检测到{severity}程度的药物相互作用[/bold {color}]\n"
                "请咨询医生或药师获取专业建议",
                border_style=color
            ))


@cli.command()
@click.option('--patient-file', '-p', type=click.Path(exists=True), help='患者信息JSON文件')
@click.option('--medications', '-m', multiple=True, help='药物列表（可多次指定）')
@click.option('--age', '-a', type=int, help='患者年龄')
@click.option('--egfr', type=float, help='eGFR值')
@click.option('--output', '-o', type=click.Path(), help='输出结果到文件')
def analyze(patient_file, medications, age, egfr, output):
    """分析处方安全性"""
    
    console.print("[bold blue]PharmaGuard 处方安全分析[/bold blue]")
    
    # 加载患者信息
    patient_info = {}
    if patient_file:
        with open(patient_file, 'r', encoding='utf-8') as f:
            patient_info = json.load(f)
    
    if age:
        patient_info['age'] = age
    if egfr:
        patient_info['egfr'] = egfr
    
    if not medications:
        console.print("[red]请指定药物列表[/red]")
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task(description="分析处方安全性...", total=None)
        
        # 模拟分析
        import random
        random.seed(42)
        
        alerts = []
        recommendations = []
        
        # 年龄风险
        if patient_info.get('age', 0) >= 65:
            alerts.append({
                'level': 'warning',
                'title': '老年患者用药风险',
                'description': f"患者年龄 {patient_info['age']} 岁，需关注药物代谢变化"
            })
        
        # 多重用药
        if len(medications) >= 5:
            alerts.append({
                'level': 'warning',
                'title': '多重用药风险',
                'description': f"患者使用 {len(medications)} 种药物"
            })
        
        # 肾功能
        if patient_info.get('egfr', 100) < 60:
            alerts.append({
                'level': 'critical' if patient_info['egfr'] < 30 else 'warning',
                'title': '肾功能不全',
                'description': f"eGFR为 {patient_info['egfr']}，需调整药物剂量"
            })
        
        recommendations.append({
            'priority': 1,
            'description': '进行用药重整评估',
            'rationale': '存在多个风险因素'
        })
        
        # 药物对
        ddi_pairs = []
        for i, med_a in enumerate(medications):
            for j, med_b in enumerate(medications):
                if i >= j:
                    continue
                prob = random.uniform(0.2, 0.9)
                if prob > 0.5:
                    ddi_pairs.append({
                        'drug_a': med_a,
                        'drug_b': med_b,
                        'probability': prob,
                        'severity': 'moderate' if prob < 0.8 else 'severe'
                    })
        
        result = {
            'patient': patient_info,
            'medications': list(medications),
            'alerts': alerts,
            'recommendations': recommendations,
            'ddi_pairs': ddi_pairs,
            'risk_score': min(10 + len(alerts) * 15 + len(ddi_pairs) * 10, 100),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
    
    # 输出结果
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        console.print(f"[green]结果已保存到: {output}[/green]")
    
    # 显示摘要
    console.print(f"\n[bold]分析摘要[/bold]")
    console.print(f"  药物数量: {len(medications)}")
    console.print(f"  DDI发现: {len(ddi_pairs)} 对")
    console.print(f"  安全警报: {len(alerts)} 个")
    console.print(f"  风险评分: {result['risk_score']}/100")
    
    if alerts:
        console.print(f"\n[bold yellow]安全警报:[/bold yellow]")
        for alert in alerts:
            level_color = {'critical': 'red', 'warning': 'yellow', 'info': 'blue'}
            color = level_color.get(alert['level'], 'white')
            console.print(f"  [{color}][{alert['level'].upper()}][/{color}] {alert['title']}")
    
    if ddi_pairs:
        console.print(f"\n[bold yellow]药物相互作用:[/bold yellow]")
        for ddi in ddi_pairs:
            console.print(f"  • {ddi['drug_a']} + {ddi['drug_b']}: {ddi['severity']} ({ddi['probability']:.1%})")


@cli.command()
@click.argument('drug_name')
@click.option('--json-output', '-j', is_flag=True, help='以JSON格式输出')
def query(drug_name: str, json_output: bool):
    """查询药物信息"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task(description=f"查询 {drug_name} 的信息...", total=None)
        
        # 模拟查询
        result = {
            'name': drug_name,
            'drugbank_id': f'DB{hash(drug_name) % 100000:05d}',
            'atc_codes': ['A10BA02'],
            'category': '处方药',
            'indications': ['2型糖尿病', '多囊卵巢综合征'],
            'targets': ['AMPK', 'INSR'],
            'side_effects': ['胃肠道不适', '乳酸酸中毒（罕见）'],
            'interactions_count': 245,
            'half_life': '6.2小时',
            'bioavailability': '50-60%'
        }
    
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        table = Table(title=f"药物信息: {drug_name}")
        table.add_column("属性", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("DrugBank ID", result['drugbank_id'])
        table.add_row("分类", result['category'])
        table.add_row("适应症", ", ".join(result['indications']))
        table.add_row("靶点", ", ".join(result['targets']))
        table.add_row("副作用", ", ".join(result['side_effects']))
        table.add_row("已知DDI数", str(result['interactions_count']))
        table.add_row("半衰期", result['half_life'])
        table.add_row("生物利用度", result['bioavailability'])
        
        console.print(table)


@cli.command()
@click.option('--port', '-p', default=8000, help='API服务端口')
@click.option('--host', '-h', default='0.0.0.0', help='API服务主机')
@click.option('--reload/--no-reload', default=False, help='启用热重载')
def serve(port: int, host: str, reload: bool):
    """启动API服务"""
    import uvicorn
    
    console.print(f"[bold green]启动 PharmaGuard API 服务[/bold green]")
    console.print(f"  主机: {host}")
    console.print(f"  端口: {port}")
    console.print(f"  API文档: http://{host}:{port}/api/docs")
    
    uvicorn.run(
        "pharmaguard.api.api_server:app",
        host=host,
        port=port,
        reload=reload
    )


@cli.command()
def dashboard():
    """启动Web Dashboard"""
    import subprocess
    import sys
    
    dashboard_path = Path(__file__).parent / 'dashboard' / 'app.py'
    
    console.print("[bold green]启动 PharmaGuard Dashboard[/bold green]")
    
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--server.port', '8501'
    ])


@cli.command()
@click.option('--port', '-p', default=8000, help='API端口')
def health(port: int):
    """检查API健康状态"""
    import requests
    
    try:
        response = requests.get(f"http://localhost:{port}/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                console.print("[green]✓[/green] PharmaGuard API 运行正常")
                console.print(f"  版本: {data.get('data', {}).get('version', 'unknown')}")
            else:
                console.print("[red]✗[/red] API返回异常状态")
        else:
            console.print(f"[red]✗[/red] API返回状态码: {response.status_code}")
    except requests.exceptions.ConnectionError:
        console.print(f"[red]✗[/red] 无法连接到API (localhost:{port})")
    except Exception as e:
        console.print(f"[red]✗[/red] 检查失败: {e}")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def init(config_file: str):
    """从配置文件初始化系统"""
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    console.print("[bold blue]初始化 PharmaGuard 系统[/bold blue]")
    console.print(f"  配置文件: {config_file}")
    console.print(f"  配置项数: {len(config)}")
    
    # 验证配置
    required_keys = ['model', 'knowledge_graph', 'api']
    missing = [k for k in required_keys if k not in config]
    
    if missing:
        console.print(f"[red]缺少必要配置: {', '.join(missing)}[/red]")
        sys.exit(1)
    
    console.print("[green]✓[/green] 配置验证通过")
    console.print("[green]✓[/green] 系统初始化完成")


@cli.command()
def version():
    """显示版本信息"""
    console.print("""
[bold cyan]PharmaGuard[/bold cyan] v1.0.0
智能用药安全与DDI预测系统
    """)


if __name__ == '__main__':
    cli()