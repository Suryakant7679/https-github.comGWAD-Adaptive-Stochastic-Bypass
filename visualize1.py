import os
import numpy as np
import matplotlib.pyplot as plt

def plot_filtered_comparison():
    stats_dir = 'stats/hist'
    if not os.path.exists(stats_dir):
        print(f"[-] Error: {stats_dir} not found.")
        return

    plt.figure(figsize=(12, 6))
    
    # 🎨 Focused Colors
    color_map = {
        'Benign': '#808080',      # Grey (Standard Behavior)
        'Camouflage': '#FF00FF'   # Magenta (Your Attack)
    }

    # Sirf ye do labels allow karenge
    target_labels = ['Benign', 'Camouflage']
    
    files = [f for f in os.listdir(stats_dir) if f.endswith('.txt')]
    found_any = False

    for file in files:
        label = file.split('_')[-1].replace('.txt', '').capitalize()
        
        if label in target_labels:
            found_any = True
            data = np.loadtxt(os.path.join(stats_dir, file), delimiter=',')
            
            if label == 'Benign':
                plt.plot(data, label='Benign (Normal User)', color=color_map[label], linewidth=3, alpha=0.5)
            else:
                plt.plot(data, label='Adaptive Camouflage (Our Attack)', color=color_map[label], linewidth=2, linestyle='--')

    if not found_any:
        print("[-] Benign or Camouflage files not found in stats/hist/")
        return

    plt.title('HoDS Pattern Mimicry: Benign vs. Adaptive Attack', fontsize=14)
    plt.xlabel('Histogram Bins (Delta Similarity)', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True, linestyle=':', alpha=0.4)
    
    plt.savefig('benign_vs_camouflage.png', dpi=300)
    print("[+] Seperate plot saved as 'benign_vs_camouflage.png'")
    plt.show()

if __name__ == "__main__":
    plot_filtered_comparison()