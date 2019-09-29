import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import collections

def get_labels(path, nb_samples):
  linear = np.zeros(nb_samples)
  i=0
  j=0
  for fold in sorted(os.listdir(path)):
    for file in os.listdir(path+"/"+fold):
      linear[j] = i
      j+=1
    i+=1 
  return linear

def get_labels_aug(path,nb_samples):
  linear = np.zeros(nb_samples)
  for idx in range(0,nb_samples,8):
    linear[idx:idx+8] = path[idx//8]
  return linear

def get_fusion_label(path,nb_samples):
  linear = np.zeros(nb_samples)
  i=0
  j=0
  while j<nb_samples:
    
    for fold in sorted(os.listdir(path)):
      for file in os.listdir(path+"/"+fold):
        linear[j] = i
        j+=1
      i+=1 
  return linear

import matplotlib as mlp
def get_pca_coef(multiples=False,
                 path=None,
                 weights=None,
                 pooling=None,
                 fakes=None,
                 fakes_path=None,
                 rotate=None,
                 rebalance=None,
                 rebalance_size=None,
                 number_of_fakes=None,
                 smote_fct=None,
                 k_neighbors=None,
                 adasyn=None,
                 stratkfold=None):
  c=[]
  x_shapes=[]
  args = dict(path=path,
    weights=cnn_weights,
    pooling=pooling_type,
    cnn_layer=cnn_layer,
    fuse=fuse,
    fusion=fusion_layers,
    rotate=rotate,
    fakes=add_fakes,
    fakes_path=fake_path,
    rebalance=rebalance_dataset,
    rebalance_size=rebalance_to,
    number_of_fakes=number_of_fakes,
    iterations=1000,
    norm=True,
    pca=True,
    dim_red=None,
    smote_fct=smote_fct,
    k_neighbors=k,
    adasyn=ADASYN_,
    stratkfold=strat_k_fold)

  for i in range(2,11,1):
    print(i)
    if multiples:
      merged=[]
      for _ in range(0,10):
        x = classify(**args,retain_info=i/10).svm_bottleneck()
        merged.append(x)
      tmp=np.mean([merged[z][1][0] for z in range(0,10)])
      print(tmp)
      x_shapes.append(X.shape[1])
      c.append(tmp)
    else:
      x = classify(**args,retain_info=i/10).svm_bottleneck()
      print(x[1][0])
      c.append(x[1][0]) 
  fig, ax = plt.subplots()
  ax.semilogy(c)
  ax.set_ylim([0,100])
  ax.yaxis.set_major_formatter(mlp.ticker.StrMethodFormatter('{x:,.0f}'))
  ax.xaxis.set_major_formatter(mlp.ticker.StrMethodFormatter('{x:,.0f}'))
  return c, x_shapes

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    


    mpl.rcParams.update({'font.size': 20})
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

   

def grid_search(path=None,weights=None,pooling='avg', dim_red=33, stratkfold=3, shuffle=True):

  feature = get_bottleneck(path=path,
                 weights=weights,
                 pooling=pooling
                )
  y = get_labels(path,feature.shape[0])
  c = []
  X_train, X_test, y_train, y_test = \
      train_test_split(feature, y, test_size=0.5, stratify=y,
                       random_state=42)
  tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 5, 10, 100, 1000], 'gamma': [1e-3, 1e-4]}]

  scores = ['precision', 'recall']

  for score in scores:
      print("# Tuning hyper-parameters for %s" % score)
      print()

      clf = GridSearchCV(SVC(), tuned_parameters, cv=3,
                         scoring='%s_macro' % score)
      clf.fit(X_train, y_train)

      print("Best parameters set found on development set:")
      print()
      print(clf.best_params_)
      print()
      print("Grid scores on development set:")
      print()
      means = clf.cv_results_['mean_test_score']
      stds = clf.cv_results_['std_test_score']
      for mean, std, params in zip(means, stds, clf.cv_results_['params']):
          print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
      print()

      print("Detailed classification report:")
      print()
      print("The model is trained on the full development set.")
      print("The scores are computed on the full evaluation set.")
      print()
      y_true, y_pred = y_test, clf.predict(X_test)
      print(classification_report(y_true, y_pred))
      print()


  
def single_class_report(y_test,y_pred):
  acc = accuracy_score(y_test,y_pred)*100
  precision_s = precision_score(y_test,y_pred,average='weighted')*100
  recall_s = recall_score(y_test,y_pred,average='weighted')*100
  f1_s = f1_score(y_test,y_pred,average='weighted')*100
  
  return [acc,precision_s,recall_s,f1_s,len(y_test)]