
import numpy as np
import matplotlib.pyplot as plt

situs = ['BANK', 'IT', 'WAIT', 'ACCOM']
FRCNN = np.asarray([40,40,40,40])
FRCNN_Focal = np.asarray([5,5,5,5])

MNET = np.asarray([40,40,40,40])
MNET_Focal = np.asarray([5,5,5,5])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('AAA')
ax1.bar(situs, FRCNN, label='F-RCNN')
ax1.bar(situs, FRCNN + FRCNN_Focal, bottom=FRCNN, label='F-RCNN + FE')
for xpos, ypos, yval in zip(situs, FRCNN+FRCNN_Focal, FRCNN_Focal):
    ax1.text(xpos, ypos, "N=%d"%yval+' %', ha="center", va="bottom")

ax2.bar(situs, MNET, label='MNET')
ax2.bar(situs, MNET + MNET_Focal, bottom=FRCNN, label='MNET')
for xpos, ypos, yval in zip(situs, MNET+MNET_Focal, MNET_Focal):
    ax1.text(xpos, ypos, "N=%d"%yval+' %', ha="center", va="bottom")

ax1.ylim(0,70)
ax2.ylim(0,70)

fig.savefig('')