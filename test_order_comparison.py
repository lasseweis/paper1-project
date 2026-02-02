
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def test_plot():
    data = []
    storylines = [
        'Multi-Model Mean',
        'Slow Jet & Northward Shift',
        'Fast Jet & Northward Shift',
        'Slow Jet & Southward Shift',
        'Fast Jet & Southward Shift',
    ]
    
    # Create dummy data
    for s in storylines:
        for _ in range(10):
            data.append({'storyline': s, 'value': np.random.randn()})
            
    df = pd.DataFrame(data)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Method 1: Target Plot (set_ylim inverted)
    ax1 = axs[0]
    sns.boxplot(data=df, y='storyline', x='value', order=storylines, ax=ax1, orient='h')
    y_limits = (len(storylines) - 0.5, -0.5)
    ax1.set_ylim(y_limits)
    ax1.set_title("Method 1: set_ylim(4.5, -0.5)")
    
    # Method 2: Figure 3 (invert_yaxis)
    ax2 = axs[1]
    sns.boxplot(data=df, y='storyline', x='value', order=storylines, ax=ax2, orient='h')
    ax2.invert_yaxis()
    ax2.set_title("Method 2: invert_yaxis()")
    
    plt.tight_layout()
    plt.savefig('test_order_comparison.png')
    print("Test plot saved to test_order_comparison.png")

if __name__ == "__main__":
    test_plot()
