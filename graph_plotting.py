# from https://stackoverflow.com/questions/66514262/plot-graphs-from-csv-file

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('log.csv')
df[['epoch', 'accuracy', 'val_accuracy']].plot(
    x='epoch',
    xlabel='x',
    ylabel='y',
    title='Accuracy VS Val_acc'
)

plt.show()