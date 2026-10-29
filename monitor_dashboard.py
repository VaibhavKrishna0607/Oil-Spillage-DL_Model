"""
Real-time Training Monitoring Dashboard
Localhost web interface to monitor training progress
"""

from flask import Flask, render_template_string, jsonify
import csv
import json
from pathlib import Path
import time

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Monitor - Oil Spill Detection</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #4CAF50;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .status-box {
            background: #2d2d2d;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background: #333;
            border-radius: 8px;
            min-width: 150px;
        }
        .metric-label {
            color: #999;
            font-size: 12px;
            text-transform: uppercase;
        }
        .metric-value {
            color: #4CAF50;
            font-size: 32px;
            font-weight: bold;
        }
        .metric-value.warn {
            color: #FF9800;
        }
        .metric-value.error {
            color: #f44336;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #2d2d2d;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        th {
            background: #333;
            color: #4CAF50;
            position: sticky;
            top: 0;
        }
        tr:hover {
            background: #333;
        }
        .best-row {
            background: #1a472a !important;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        .status-running {
            background: #4CAF50;
            color: white;
        }
        .status-stopped {
            background: #f44336;
            color: white;
        }
        .chart-container {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Training Monitor - Oil Spill Detection Model</h1>
        
        <div class="status-box">
            <h2>Training Status: <span class="status-badge status-{{ status }}">{{ status.upper() }}</span></h2>
            <p><strong>Model:</strong> {{ model_path }}</p>
            <p><strong>Started:</strong> {{ start_time }}</p>
            <p><strong>Elapsed:</strong> {{ elapsed }}</p>
            <p><strong>Last Update:</strong> {{ last_update }}</p>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ progress }}%;">
                Epoch {{ current_epoch }}/{{ total_epochs }} ({{ progress }}%)
            </div>
        </div>

        <div style="margin: 30px 0;">
            <div class="metric">
                <div class="metric-label">Current Val Dice</div>
                <div class="metric-value">{{ current_dice }}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Val Dice</div>
                <div class="metric-value">{{ best_dice }}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Baseline</div>
                <div class="metric-value">{{ baseline_dice }}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Improvement</div>
                <div class="metric-value {{ 'warn' if improvement < 5 else '' }}">+{{ improvement }}%</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>📊 Epoch History</h2>
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
                    {% for row in history %}
                    <tr class="{{ 'best-row' if row.is_best else '' }}">
                        <td>{{ row.epoch }} {{ '★' if row.is_best else '' }}</td>
                        <td>{{ "%.4f"|format(row.train_loss) }}</td>
                        <td>{{ "%.2f"|format(row.train_dice * 100) }}%</td>
                        <td>{{ "%.4f"|format(row.val_loss) }}</td>
                        <td>{{ "%.2f"|format(row.val_iou * 100) }}%</td>
                        <td>{{ "%.2f"|format(row.val_dice * 100) }}%</td>
                        <td>{{ "%.2f"|format(row.val_threshold) }}</td>
                        <td>{{ "%.2e"|format(row.learning_rate) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="status-box">
            <h3>💡 Training Configuration</h3>
            <p><strong>Batch Size:</strong> {{ config.batch_size }}</p>
            <p><strong>Initial LR:</strong> {{ config.lr }}</p>
            <p><strong>Total Epochs:</strong> {{ config.epochs }}</p>
            <p><strong>Patience:</strong> {{ config.patience }}</p>
        </div>

        <p style="text-align: center; color: #666; margin-top: 40px;">
            Auto-refreshes every 10 seconds | 
            <a href="/" style="color: #4CAF50;">Refresh Now</a>
        </p>
    </div>
</body>
</html>
"""


def read_latest_results():
    """Read the latest training results"""
    checkpoint_dir = ROOT / "ml_model" / "checkpoints"
    
    # Try to read continuation history
    history_file = checkpoint_dir / "dataset2_continued_history.csv"
    meta_file = checkpoint_dir / "dataset2_continued_meta.json"
    
    if not history_file.exists():
        return None
    
    # Read history CSV
    history = []
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
                    'is_best': False
                })
    except Exception as e:
        print(f"Error reading history: {e}")
        return None
    
    # Read metadata
    meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
        except:
            pass
    
    # Mark best epoch
    if history:
        best_dice = max(h['val_dice'] for h in history)
        for h in history:
            if h['val_dice'] == best_dice:
                h['is_best'] = True
                break
    
    return {
        'history': history,
        'meta': meta
    }


@app.route('/')
def index():
    """Main dashboard page"""
    results = read_latest_results()
    
    if not results:
        return """
        <html>
        <body style="background: #1e1e1e; color: white; font-family: Arial; text-align: center; padding: 50px;">
            <h1>⏳ Waiting for training to start...</h1>
            <p>No training data found yet. Training will appear here once started.</p>
            <p><a href="/" style="color: #4CAF50;">Refresh</a></p>
        </body>
        </html>
        """
    
    history = results['history']
    meta = results['meta']
    
    # Calculate metrics
    current_epoch = len(history)
    best_dice = max(h['val_dice'] for h in history) if history else 0
    current_dice = history[-1]['val_dice'] if history else 0
    baseline_dice = meta.get('baseline_dice', 0.53)
    improvement = ((best_dice - baseline_dice) / baseline_dice * 100) if baseline_dice > 0 else 0
    
    # Progress
    total_epochs = 20  # From command
    progress = (current_epoch / total_epochs) * 100
    
    # Time estimates
    elapsed_min = meta.get('elapsed_minutes', 0)
    elapsed = f"{int(elapsed_min)} minutes" if elapsed_min < 60 else f"{elapsed_min/60:.1f} hours"
    
    return render_template_string(
        HTML_TEMPLATE,
        status='running' if current_epoch < total_epochs else 'stopped',
        model_path='dataset2_continued.pth',
        start_time='Recent',
        elapsed=elapsed,
        last_update=time.strftime('%Y-%m-%d %H:%M:%S'),
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        progress=int(progress),
        current_dice=f"{current_dice*100:.2f}",
        best_dice=f"{best_dice*100:.2f}",
        baseline_dice=f"{baseline_dice*100:.2f}",
        improvement=f"{improvement:.1f}",
        history=reversed(history),  # Show latest first
        config={
            'batch_size': 8,
            'lr': '2e-5',
            'epochs': 20,
            'patience': 10
        }
    )


@app.route('/api/status')
def api_status():
    """API endpoint for status"""
    results = read_latest_results()
    if not results:
        return jsonify({'status': 'waiting', 'message': 'No training data yet'})
    
    history = results['history']
    meta = results['meta']
    
    return jsonify({
        'status': 'running',
        'current_epoch': len(history),
        'total_epochs': 20,
        'best_dice': max(h['val_dice'] for h in history) if history else 0,
        'current_dice': history[-1]['val_dice'] if history else 0,
        'baseline_dice': meta.get('baseline_dice', 0.53),
    })


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Training Monitor Dashboard Starting...")
    print("=" * 80)
    print(f"\n📊 Dashboard URL: http://localhost:5000")
    print(f"\n💡 Open your browser and navigate to: http://localhost:5000")
    print(f"\n⏱️  Auto-refreshes every 10 seconds")
    print(f"\n🛑 Press Ctrl+C to stop\n")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
