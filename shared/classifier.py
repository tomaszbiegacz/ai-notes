import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, ConfusionMatrixDisplay

def precisionRecall_check(model, X, y, cv=None):
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    print(confusion_matrix(y, y_pred))
    print()
    print(f'precision: {precision_score(y, y_pred)}')
    print(f'recall:    {recall_score(y, y_pred)}')
    print(f'f1:        {f1_score(y, y_pred)}')

def precisionRecall_curve(model, X, y, y_scores, figs_size=(4, 4)):
  precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
  _, axs = plt.subplots(ncols=2, nrows=1, figsize=(figs_size[0] * 2, figs_size[1]))

  ax = axs[0]
  ax.plot(thresholds, precisions[:-1], "b--", label="precision")
  ax.plot(thresholds, recalls[:-1], "g--", label="recall")
  ax.grid()
  ax.set_xlabel("threshold")
  ax.legend()

  ax = axs[1]
  ax.plot(recalls, precisions)
  ax.grid()
  ax.set_xlabel("recall")
  ax.set_ylabel("precision")

  plt.show()

def recallFallout_curve(model, X, y, y_scores, figs_size=(4, 4)):
  fpr, tpr, _ = roc_curve(y, y_scores)

  plt.figure(figsize=figs_size)
  plt.plot(fpr, tpr)
  plt.title("roc")

  plt.show()
  print(f'Area under: {roc_auc_score(y, y_scores)}')

def confusionMatrix_from_predictions(y, y_pred):
    _, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
    axs[0].set_title("Confusion matrix")
    ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize="true", ax=axs[0], values_format=".0%")   
    axs[1].set_title("Classification errors distribution by row")
    ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize="true", sample_weight=(y != y_pred), ax=axs[1], values_format=".0%")
    axs[2].set_title("Classification errors distribution by col")
    ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize="pred", sample_weight=(y != y_pred), ax=axs[2], values_format=".0%")
