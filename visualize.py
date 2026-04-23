import os
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison():
    stats_dir = 'stats/hist'
    if not os.path.exists(stats_dir):
        print(f"[-] Error: {stats_dir} not found.")
        return

    plt.figure(figsize=(14, 7))
    
    # 🎨 Color Palette for Clarity
    # Benign ko hum 'Black' ya 'Gray' rakhenge base ki tarah
    # Camouflage ko 'Bright Green' taaki wo alag chamke
    color_map = {
        'Benign': '#000000',      # Black (Solid Base)
        'Camouflage': '#FF00FF',  # Magenta (High Contrast)
        'Hsja': '#1f77b4',        # Blue
        'Nes': '#d62728',         # Red
        'Sign-flip': '#ff7f0e',   # Orange
        'Adaptive': '#2ca02c'     # Green
    }

    files = [f for f in os.listdir(stats_dir) if f.endswith('.txt')]
    
    for file in files:
        label = file.split('_')[-1].replace('.txt', '').capitalize()
        data = np.loadtxt(os.path.join(stats_dir, file), delimiter=',')
        
        color = color_map.get(label, None)
        
        # Plotting Logic
        if label == 'Benign':
            # Benign ko thoda mota aur niche rakhenge
            plt.plot(data, label=label, color=color, linewidth=3, alpha=0.4, zorder=1)
        elif label == 'Camouflage':
            # Camouflage ko dashed ya bright color mein upar rakhenge
            plt.plot(data, label=label, color=color, linewidth=2, linestyle='--', zorder=10)
        else:
            plt.plot(data, label=label, color=color, linewidth=1.5, alpha=0.8, zorder=5)

    plt.title('Normalized GWAD Histogram Comparison (High Visibility Mode)', fontsize=14)
    plt.xlabel('Histogram Bins (Delta Similarity)', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig('clear_bypass_plot.png', dpi=300)
    print("[+] Clear plot saved as 'clear_bypass_plot.png'")
    plt.show()

if __name__ == "__main__":
    plot_comparison()