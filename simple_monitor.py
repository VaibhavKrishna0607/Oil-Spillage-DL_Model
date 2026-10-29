"""
Simple Training Monitor - Creates HTML file that auto-refreshes
No dependencies needed - just open monitor.html in browser
"""

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def generate_html():
    """Generate monitoring HTML page"""
    checkpoint_dir = ROOT / "ml_model" / "checkpoints"
    history_file = checkpoint_dir / "dataset2_continued_history.csv"
    meta_file = checkpoint_dir / "dataset2_continued_meta.json"
    
    # Read history
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    history.append({
                        'epoch': int(row['epoch']),
                        'train_loss': float(row['train_loss']),
                        'train_dice': float(row['train_dice']),
                        'val_loss': float(row['val_loss']),
                        'val_iou': float(row['val_iou']),
                        'val_dice': float(row['val_dice']),
                        'val_threshold': float(row['val_threshold']),
                        'learning_rate': float(row['learning_rate']),
                    })
        except Exception as e:
            print(f"Error reading history: {e}")
    
    # Read metadata
    meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
        except:
            pass
    
    # Calculate metrics
    current_epoch = len(history)
    best_dice = max(h['val_dice'] for h in history) if history else 0
    current_dice = history[-1]['val_dice'] if history else 0
    baseline_dice = meta.get('baseline_dice', 0.4874)
    improvement = ((best_dice - baseline_dice) / baseline_dice * 100) if baseline_dice > 0 else 0
    
    # Find best epoch
    best_epoch = 0
    if history:
        for h in history:
            if h['val_dice'] == best_dice:
                best_epoch = h['epoch']
                break
    
    # Progress
    total_epochs = 20
    progress = (current_epoch / total_epochs) * 100 if total_epochs > 0 else 0
    
    # Generate table rows
    table_rows = ""
    for h in reversed(history):  # Latest first
        is_best = (h['val_dice'] == best_dice)
        row_class = 'best-row' if is_best else ''
        best_mark = ' ★' if is_best else ''
        
        table_rows += f"""
        <tr class="{row_class}">
            <td>{h['epoch']}{best_mark}</td>
            <td>{h['train_loss']:.4f}</td>
            <td>{h['train_dice']*100:.2f}%</td>
            <td>{h['val_loss']:.4f}</td>
            <td>{h['val_iou']*100:.2f}%</td>
            <td>{h['val_dice']*100:.2f}%</td>
            <td>{h['val_threshold']:.2f}</td>
            <td>{h['learning_rate']:.2e}</td>
        </tr>
        """
    
    elapsed_min = meta.get('elapsed_minutes', 0)
    elapsed = f"{int(elapsed_min)} min" if elapsed_min < 60 else f"{elapsed_min/60:.1f} hrs"
    
    status = 'RUNNING' if current_epoch < total_epochs else 'COMPLETED'
    status_class = 'status-running' if status == 'RUNNING' else 'status-stopped'
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Monitor - Oil Spill Detection</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #4CAF50;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .status-box {{
            background: #2d2d2d;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background: #333;
            border-radius: 8px;
            min-width: 150px;
        }}
        .metric-label {{
            color: #999;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metric-value {{
            color: #4CAF50;
            font-size: 32px;
            font-weight: bold;
        }}
        .metric-value.warn {{
            color: #FF9800;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #2d2d2d;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }}
        th {{
            background: #333;
            color: #4CAF50;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #333;
        }}
        .best-row {{
            background: #1a472a !important;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        .status-running {{
            background: #4CAF50;
            color: white;
        }}
        .status-stopped {{
            background: #f44336;
            color: white;
        }}
        .chart-container {{
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Training Monitor - Oil Spill Detection Model</h1>
        
        <div class="status-box">
            <h2>Training Status: <span class="status-badge {status_class}">{status}</span></h2>
            <p><strong>Model:</strong> dataset2_continued.pth (Option C - Aggressive)</p>
            <p><strong>Config:</strong> Batch=8, LR=2e-5, Patience=10</p>
            <p><strong>Elapsed:</strong> {elapsed}</p>
            <p><strong>Last Update:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress:.0f}%;">
                Epoch {current_epoch}/{total_epochs} ({progress:.0f}%)
            </div>
        </div>

        <div style="margin: 30px 0;">
            <div class="metric">
                <div class="metric-label">Current Val Dice</div>
                <div class="metric-value">{current_dice*100:.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Val Dice</div>
                <div class="metric-value">{best_dice*100:.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Baseline</div>
                <div class="metric-value">{baseline_dice*100:.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Improvement</div>
                <div class="metric-value {'warn' if improvement < 5 else ''}">+{improvement:.1f}%</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>📊 Epoch History ({len(history)} epochs completed)</h2>
            {f'<p style="color: #666;">Waiting for first epoch to complete...</p>' if not history else f'''
            <table>
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Train Loss</th>
                        <th>Train Dice</th>
                        <th>Val Loss</th>
                        <th>Val IoU</th>
                        <th>Val Dice</th>
                        <th>Threshold</th>
                        <th>LR</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            '''}
        </div>

        <div class="status-box">
            <h3>🎯 Target: 95%+ Val Dice for deployment</h3>
            <p><strong>Current Progress:</strong> {current_dice*100:.2f}% → Target: 95%</p>
            <p><strong>Remaining:</strong> {95 - current_dice*100:.2f} percentage points</p>
        </div>

        <p style="text-align: center; color: #666; margin-top: 40px;">
            Auto-refreshes every 5 seconds
        </p>
    </div>
</body>
</html>
    """
    
    return html


def main():
    """Generate monitor HTML continuously"""
    output_file = ROOT / "monitor.html"
    
    print("=" * 80)
    print("🚀 Simple Training Monitor")
    print("=" * 80)
    print(f"\n📊 Generating monitor.html every 5 seconds...")
    print(f"\n💡 Open in browser: file:///{output_file.as_posix()}")
    print(f"\n🛑 Press Ctrl+C to stop\n")
    print("=" * 80)
    
    try:
        while True:
            html = generate_html()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✓ Updated at {time.strftime('%H:%M:%S')}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n✓ Monitor stopped")


if __name__ == '__main__':
    main()
