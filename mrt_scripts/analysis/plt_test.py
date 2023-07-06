import numpy as np
import matplotlib.pyplot as plt

plt.plot([2, 3, 8, 3])
plt.save_fig('checkpoints/wmt14_en2de_uncased_bleurt_beam12_lr5e-4/plot_test.jpg')

# python3 mrt_scripts/analysis/plt_test.py