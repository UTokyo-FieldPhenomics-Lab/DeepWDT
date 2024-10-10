import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def create_evaluation_graphs(df, metrics, output_path):
    fig = make_subplots(rows=3, cols=3, subplot_titles=[metric.replace('_', ' ').title() for metric in metrics])

    versions = df['version'].unique()
    color_maps = ['Blues', 'Greens', 'Oranges', 'Reds']

    for i, metric in enumerate(metrics):
        row = i // 3 + 1
        col = i % 3 + 1

        for idx, version in enumerate(versions):
            version_data = df[df['version'] == version]
            k_values = sorted(version_data['K'].unique())
            cmap = plt.get_cmap(color_maps[idx % len(color_maps)])

            for k_idx, K in enumerate(k_values):
                data = version_data[version_data['K'] == K]
                color = mcolors.rgb2hex(cmap((k_idx + 1) / len(k_values)))

                fig.add_trace(
                    go.Scatter(x=data['epoch'], y=data[metric], mode='lines+markers',
                               name=f'{version}, K={K}', line=dict(color=color)),
                    row=row, col=col
                )

        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)

    fig.update_layout(height=1000, width=1200, title_text="Evaluation Metrics")
    fig.write_html(output_path)

# Load the data
df = pd.read_csv('runs/evaluation/tracking/eval_track.csv')

# Define metrics
metrics = ['detections_percentage', 'angle_mae', 'angle_rmse', 'angle_r2', 'angle_bias',
           'duration_mae', 'duration_rmse', 'duration_r2', 'duration_bias']

# Create directory if it doesn't exist
save_dir = 'runs/evaluation/tracking'
os.makedirs(save_dir, exist_ok=True)

# Create and save the graph
save_path = os.path.join(save_dir, 'evaluation_metrics.html')
create_evaluation_graphs(df, metrics, save_path)