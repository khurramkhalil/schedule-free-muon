import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

def set_style():
    """Set publication-quality style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['savefig.dpi'] = 300

def plot_run(df, run_name, color, output_path):
    """Plot training and validation loss for a single run."""
    plt.figure()
    
    # Plot Training Loss
    sns.lineplot(data=df, x='step', y='train_loss', label='Train Loss', color=color, alpha=0.3)
    
    # Calculate and plot smoothed training loss
    df['train_loss_smooth'] = df['train_loss'].rolling(window=20, min_periods=1).mean()
    sns.lineplot(data=df, x='step', y='train_loss_smooth', label='Train Loss (Smoothed)', color=color)
    
    # Plot Validation Loss (scatter points)
    val_data = df.dropna(subset=['val_loss'])
    if not val_data.empty:
        plt.scatter(val_data['step'], val_data['val_loss'], color='red', marker='x', s=100, label='Val Loss', zorder=5)
    
    plt.title(f'{run_name} Training Dynamics')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_comparison(adam_df, muon_df, output_path):
    """Plot comparison of smoothed training loss."""
    plt.figure()
    
    # Smooth data
    adam_df['train_loss_smooth'] = adam_df['train_loss'].rolling(window=20, min_periods=1).mean()
    muon_df['train_loss_smooth'] = muon_df['train_loss'].rolling(window=20, min_periods=1).mean()
    
    sns.lineplot(data=adam_df, x='step', y='train_loss_smooth', label='AdamW', color='tab:blue')
    sns.lineplot(data=muon_df, x='step', y='train_loss_smooth', label='SF-Muon', color='tab:orange')
    
    # Plot Validation dots if available
    adam_val = adam_df.dropna(subset=['val_loss'])
    muon_val = muon_df.dropna(subset=['val_loss'])
    
    if not adam_val.empty:
        plt.scatter(adam_val['step'], adam_val['val_loss'], color='tab:blue', marker='o', s=50, label='AdamW Val')
    if not muon_val.empty:
        plt.scatter(muon_val['step'], muon_val['val_loss'], color='tab:orange', marker='^', s=50, label='SF-Muon Val')

    plt.title('AdamW vs SF-Muon Convergence')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def main():
    os.makedirs('figures', exist_ok=True)
    set_style()
    
    # Load Data
    try:
        adam_df = pd.read_csv('logs/adamw_run.csv')
        muon_df = pd.read_csv('logs/sf_muon_run.csv')
    except FileNotFoundError as e:
        print(f"Error loading logs: {e}")
        return

    # Plot AdamW
    plot_run(adam_df, 'AdamW', 'tab:blue', 'figures/adamw_training.png')
    
    # Plot SF-Muon
    plot_run(muon_df, 'SF-Muon', 'tab:orange', 'figures/sf_muon_training.png')
    
    # Plot Comparison
    plot_comparison(adam_df, muon_df, 'figures/comparison.png')

if __name__ == "__main__":
    main()
