import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


config = {
    "font.family":'Times New Roman', # 设置字体类型
    "font.size": 14.5,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)


def img_hist(img, hist_save_path, tilte='Ours'):
     
     mu = np.mean(img)
     sigma = np.std(img)

     fig, ax = plt.subplots(figsize = (5, 4))
     bins = np.array([i for i in range(-90, 90, 1)]) 

     # ax.hist(img, bins, label='resi', density = True, alpha = 1.0, color='deepskyblue', ec="dodgerblue") 
     # ax.hist(img, bins, label='resi', density = True, alpha = 1.0, color='dodgerblue') 
     ax.hist(img, bins, label='resi', density = True, alpha = 1.0,) 
     # ax.hist(img, bins, label='resi', density = True, alpha = 1.0, color='deepskyblue', ec="orange") 

    # add a 'best fit' line
     y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
          np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
     # ax.plot(bins, y, '--', color='firebrick')
     ax.plot(bins, y, '--')
     ax.set_xlabel('Pixels Value')
     ax.set_ylabel('Probability density')
     ax.set_title(tilte)
     
     # ax.set_title('Distribution of the residual between '
     #      fr'$S$ and $S$')


     fig.tight_layout()
     plt.savefig(hist_save_path)



