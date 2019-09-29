from feat_extract import *
from classify import *
import warnings
warnings.filterwarnings('ignore')

trait_variation = 0.9
#cnn_weights= 'weights/pretrained.h5'
cnn_weights= 'weights/model224nobg-body_global.h5'
#cnn_weights= 'weights/4shizo-body.h5'
cnn_layer = 5
pooling_type = 'avg'
strat_k_fold = 2
ADASYN_ = False 
k = 6 

path = 'datasets/pleo224'
#path = 'datasets/shizo224nd_7class'
rotate = False 
rotation_degree = 20 
add_fakes = False 
fake_path= 'fakes4_5class'
number_of_fakes = 20
rebalance_dataset = False 
rebalance_to = 100 
SMOTE_fct = False 
print_results = True 
display_dataset_table = False
multiples = None
fusion_layers = [3,4]
fuse = False

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
      retain_info=trait_variation,
      dim_red=None,
      smote_fct=SMOTE_fct,
      k_neighbors=k,
      adasyn=ADASYN_,
      stratkfold=strat_k_fold,
      repeated=multiples)

results = classify(**args).svm_bottleneck()

print("%.2f" % results[1][0] )
print("& %.2f" % results[1][1])
print("& %.2f" % results[1][2])
print("& %.2f" % results[1][3])
print("& %.2f" % results[1][4])


c, x_shape = get_pca_coef(multiples=multiples,
                 path=path,
                 weights=cnn_weights,
                 pooling=pooling_type,
                 smote_fct=SMOTE_fct,
                 rotate=rotate,
                 fakes=add_fakes,
                 fakes_path=fake_path,
                 rebalance=rebalance_dataset,
                 rebalance_size=rebalance_to,
                 number_of_fakes=number_of_fakes,
                 k_neighbors=k,
                 adasyn=ADASYN_,
                 stratkfold=strat_k_fold)
print(c,x_shape)