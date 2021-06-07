import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import csv
from sklearn import metrics
import pandas as pd
import io
import base64

class Plotting():
    def __init__(self,metrics):
        self.metrics = metrics

    def fig_to_base64(self,fig):
        img = io.BytesIO()
        fig.savefig(img, format='png',
                    bbox_inches='tight')
        img.seek(0)

        return base64.b64encode(img.getvalue())

    def plot_confusion_matrix(self):
        conf_matrix = self.metrics["confusion_matrix"]
        no_of_classes = len(conf_matrix)
        classes=[]
        for i in range(0, no_of_classes):
            classes.append(str(i))
        df_cfm = pd.DataFrame(conf_matrix, index = classes, columns = classes)
        plt.figure(figsize = (10,7))
        cfm_plot = sn.heatmap(df_cfm, annot=True)
        # cfm_plot.figure.savefig("cfm.png")

        encoded = self.fig_to_base64(cfm_plot.figure)
        # my_html = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
        return encoded.decode('utf-8')

    def plot_roc_auc(self):
        pass

    def plot_metrics(self):
        pass

    def plot_feature_importance(self):
        pass

    def plot_prc(self):
        pass
