import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_and_save_graph(df, x_col, y_col, title, output_path):
    fig = go.Figure()
    versions = df['version'].unique()
    color_maps = ['Blues', 'Greens', 'Oranges', 'Reds']
    
    for idx, version in enumerate(versions):
        version_data = df[df['version'] == version]
        k_values = sorted(version_data['K'].unique())
        cmap = plt.get_cmap(color_maps[idx % len(color_maps)])
        
        for k_idx, K in enumerate(k_values):
            data = version_data[(version_data['K'] == K) & (version_data['split'] == 'val')]
            color = mcolors.rgb2hex(cmap(k_idx / (len(k_values) - 1)))
            fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], mode='lines+markers', 
                                     name=f'{version}, K={K}', line=dict(color=color)))
    
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.write_html(output_path)

evalf_df = pd.read_csv('runs/evaluation/recognition/evalf.csv')
create_and_save_graph(evalf_df, 'epoch', 'mAP', 'mAP vs Epoch (evalf)', 'runs/evaluation/recognition/evalf_graph.html')

evalv_df = pd.read_csv('runs/evaluation/recognition/evalv.csv')
create_and_save_graph(evalv_df, 'epoch', '0.3', 'mAP vs Epoch (evalv)', 'runs/evaluation/recognition/evalv_graph.html')