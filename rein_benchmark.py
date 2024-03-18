import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import csv
import sys
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import cleaners as cln
import numpy as np
import argparse
import pandas as pd

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from datasets import get_tf_database, load_regression, load_regression_dirty, load_features_and_data, reverse_categorical_columns, reverse_to_input_domain
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x, include_errors_at_random
from utils import create_and_train_LOP, create_and_train_classifier
from error_detection import predict_on_enhanced, eval_correctly, eval_numeric_rmse
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from copy import deepcopy

parser = argparse.ArgumentParser(description="Disentangled Latent Space Operator for Data Engineering")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--latent", type=int, default=240)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--dataset", default='adult')
parser.add_argument("--experiment", default='vs_dirty')
parser.add_argument("--eval_tuples", type=int, default=-1)

args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)
print("Is Eager execution?",tf.executing_eagerly())
LOP = None

@tf.function
def _translate_1(inputs):
    zs  = inputs
    column = 0
    for i in range(zs.shape[0]):
        new_z = LOP.translate_operator(zs[column], 1)
        zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
        column = column + 1
    return zs

def generate_cleaned_data(x_, Zs, Ks, decoder):
    #avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_, xs) 
    return x_p


#CONFIGS========================================================
ds = args.dataset
T_EXAMPLES = args.eval_tuples
V_EXAMPLES = args.eval_tuples 
T_EVAL_EXAMPLES = args.eval_tuples
V_EVAL_EXAMPLES = args.eval_tuples
MODEL_EPOCHS = 200
K_EPOCHS = args.epochs
MISSING_REPLACE = '3.0'
T = 'missing_values'
K = args.K

#LOAD AIRBNB, NASA or BIKE=====================================
start = time.time()

#MAX AND MIN FOR THE ALREADY FILTERED SET + TARGET
x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, True, normalize_sklearn = True)
x_dirty_train, y_dirty_train, x_test, y_test = load_regression_dirty(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)

print(y_test.shape)
COLS = x_clean_train.shape[1]
print("# COLUMNS:", COLS)

train_dataset = get_tf_database(x_clean_train[0:T_EXAMPLES],
                                x_clean_train[0:T_EXAMPLES],
                                args.batch_size)

val_dataset = get_tf_database(x_clean[0:V_EXAMPLES],
                              x_clean[0:V_EXAMPLES],
                              args.batch_size)
print(time.time() - start)


# TRAIN MODELS========================================================
start = time.time()
encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_dataset,
                                                           COLS,
                                                           args.latent,
                                                           K,
                                                           1,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
print("LOP ",time.time() - start, "sec", "Train loss:", t_acc," VAL loss:" , v_acc)

for el in encoder.layers:
     el.trainable = False
for el in decoder.layers:
    el.trainable = False

   
#Default Classifier MODEL========================================================================
t_dataset = get_tf_database(x_clean_train, y_clean_train, args.batch_size)
v_dataset = get_tf_database(x_clean, y_clean, args.batch_size)    

start = time.time()
nnreg_model = create_and_train_classifier(t_dataset,
                                          v_dataset,
                                          COLS,
                                          args.latent,
                                          T,
                                          n_epochs=MODEL_EPOCHS,
                                          model_name=args.dataset)
print("NN train ", time.time() - start, "sec")



#Enhanced MODEL===============================================================================
concatenation_model = Concatenate()(decoder.output)
enhanced_model = Model(inputs=decoder.input, outputs= nnreg_model(concatenation_model))
enhanced_model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

n_scores = []
en_scores = []
en_smart_scores = []
clean_scores = []

#MISSING P percentage of values==================================================
x_test = x_test[0:V_EXAMPLES] 
y_test = y_test[0:V_EXAMPLES]
print("# tuples for test:", x_test.shape[0])

percentages_of_noise = [0.0]

# clean_scores.append(nnreg_model.evaluate(x_clean, y_clean)[1])
# n_scores.append(nnreg_model.evaluate(x_test, y_test)[1]) #dirty score
# Zs, Ks = predict_on_enhanced(x_test, LOP, encoder, decoder, _translate_1)
# en_scores.append(enhanced_model.evaluate(tf.unstack(Zs, axis = 1), y_test)[1])
# detections = eval_correctly(x_test, y_test, Zs, Ks, decoder, nnreg_model)
# en_smart_scores.append(detections)
# #eval_correctly(x_test, y_test, Zs, Ks, decoder, nnreg_model)



############ WRITE CLEAN DATA #################
headers,  target_name, dirty_data, _, data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)
full_header = headers["full_header"]
header_with_y = headers["filtered_header_with_y"]
filtered_header = headers["filtered_header"]
numeric_header = headers["numeric_header"] 
categorical_header = headers["categorical_header"]

X_dirty = deepcopy(data_with_y[filtered_header]).to_numpy()
Zs_csv, Ks_csv = predict_on_enhanced(X_dirty, LOP, encoder, decoder, _translate_1)
clean_csv = generate_cleaned_data(X_dirty, Zs_csv, Ks_csv, decoder)


#TODO refine predicts twice%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Zs_csv, Ks_csv = predict_on_enhanced(clean_csv, LOP, encoder, decoder, _translate_1)
# clean_csv = generate_cleaned_data(clean_csv, Zs_csv, Ks_csv, decoder)

# Zs_csv, Ks_csv = predict_on_enhanced(clean_csv, LOP, encoder, decoder, _translate_1)
# clean_csv = generate_cleaned_data(clean_csv, Zs_csv, Ks_csv, decoder)

# Zs_csv, Ks_csv = predict_on_enhanced(clean_csv, LOP, encoder, decoder, _translate_1)
# clean_csv = generate_cleaned_data(clean_csv, Zs_csv, Ks_csv, decoder)

# Zs_csv, Ks_csv = predict_on_enhanced(clean_csv, LOP, encoder, decoder, _translate_1)
# clean_csv = generate_cleaned_data(clean_csv, Zs_csv, Ks_csv, decoder)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#GENERATE THE RMSE COMPARISION TO REIN#######################################
#print("C ", dirty_data.iloc[np.where(dirty_data.isna())[0]]) #NaN errors
rmse_dirty = mean_squared_error(clean_data[numeric_header], dirty_data[numeric_header], squared = False)
#print('Dirty RMSE: ', rmse_dirty) #squared False return RMSE


#EVAL NUMERIC RMSE###########################################################
df_cleaned = deepcopy(dirty_data)
df_cleaned[filtered_header] = clean_csv
rmse = mean_squared_error(clean_data[numeric_header], df_cleaned[numeric_header], squared = False)
#print('Encoder-decoder RMSE: ', rmse) #squared False return RMSE


def _clean_with_error_detector(_df_cleaned, _dirty_data, _detections):
    rows_ = _detections[:, 0] - 2 # rein index starts at 1 and jumps the header
    cols_ = _detections[:, 1]
    df_ = deepcopy(_dirty_data) #clean_data)
    #only use know dirty position repairs
    for r, c in zip(rows_, cols_):
        df_.iloc[r, c] =  _df_cleaned.iloc[r, c]
    return df_

#with ed2 detections -------------------------------------
rmse_ed2 = 0
# ed2_detections = pd.read_csv(f'./DATASETS_REIN/{args.dataset}/ed2/detections.csv').to_numpy()
# df_ed2 = _clean_with_error_detector(df_cleaned, dirty_data, ed2_detections)
# rmse_ed2 = mean_squared_error(clean_data[numeric_header], df_ed2[numeric_header], squared = False)
#print('ED2 detections RMSE: ', rmse_ed2)

#with dboost detections -------------------------------------
rmse_dboost = 0
# dboost_detections = pd.read_csv(f'./DATASETS_REIN/{args.dataset}/dboost/detections.csv').to_numpy()
# df_dboost = _clean_with_error_detector(df_cleaned, dirty_data, dboost_detections)
# rmse_dboost = mean_squared_error(clean_data[numeric_header], df_dboost[numeric_header], squared = False)
#print('DBOOST detections RMSE: ', rmse_dboost)


#with ground truth detections -------------------------------------
rmse_gt = 0
# gt_detections = pd.read_csv(f'./DATASETS_REIN/{args.dataset}/actual_errors01.csv').to_numpy()
# df_gt_cleaned = _clean_with_error_detector(df_cleaned, dirty_data, gt_detections)
# rmse_gt = mean_squared_error(clean_data[numeric_header], df_gt_cleaned[numeric_header], squared = False)
#print('GT detections RMSE: ', rmse_gt)



#EVAL NUMERIC RMSE BY COLUMN################################################
rmse_by_column_dirty = 0
rmse_by_column_clean = 0
rmse_by_column_gt = 0
n_num_cols = len(numeric_header) #len(numeric_header) + 1 #rein benchmark includes the target with 0 errors
if n_num_cols > 0 :
    print("\n  RMSE (less is better) on Numerical Columns Dirty vs Clean")
    for num_col in numeric_header:
        r_clean = mean_squared_error(clean_data[num_col], df_cleaned[num_col], squared = False)
        r_dirty = mean_squared_error(clean_data[num_col], dirty_data[num_col], squared = False)
        #r_gt = mean_squared_error(clean_data[num_col], df_gt_cleaned[num_col], squared = False)
        print(f'{num_col}', r_dirty, " vs ", r_clean)
        rmse_by_column_dirty += r_dirty
        rmse_by_column_clean += r_clean
        #rmse_by_column_gt += r_gt

    rmse_by_column_clean =  rmse_by_column_clean / n_num_cols
    rmse_by_column_dirty =  rmse_by_column_dirty / n_num_cols
    #rmse_by_column_gt =  rmse_by_column_gt / n_num_cols

    print(f'TOTAL: {rmse_by_column_dirty} vs {rmse_by_column_clean}')#  vs {rmse_by_column_gt}')



#EVAL CATEGORICAL ACCURACY BY COLUMN###################################################
#print(clean_data.head())

c = reverse_to_input_domain(args.dataset, clean_data, FULL_SCALER, CAT_ENCODER)
d = reverse_to_input_domain(args.dataset, dirty_data, FULL_SCALER, CAT_ENCODER)
l = reverse_to_input_domain(args.dataset, df_cleaned, FULL_SCALER, CAT_ENCODER)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
#print(c.head())
#print("C ", l.iloc[np.where(l.isna())[0]]) #NaN errors

#dbo = reverse_to_input_domain(args.dataset, df_dboost, FULL_SCALER, CAT_ENCODER)
#ed2 = reverse_to_input_domain(args.dataset, df_ed2, FULL_SCALER, CAT_ENCODER)
#gtd = reverse_to_input_domain(args.dataset, df_gt_cleaned, FULL_SCALER, CAT_ENCODER)

accuracy_categories_dirty = 0
accuracy_categories = 0
accuracy_categories_gt_detections = 0
accuracy_categories_dboost_detections = 0
accuracy_categories_ed2_detections = 0
n_cat_cols = len(categorical_header) #- 1 #beer_name
avg_type = 'micro'

if n_cat_cols > 0:
    print("\n  Accuracy (more is better) on Categorical Columns Dirty vs Clean")
    for cat_col in categorical_header:
        acc_dirty = f1_score(c[cat_col], d[cat_col], average = avg_type)
        acc_clean = f1_score(c[cat_col], l[cat_col], average = avg_type)
        #acc_dboost_detections = f1_score(c[cat_col], dbo[cat_col], average = avg_type)
        #acc_ed2_detections = f1_score(c[cat_col], ed2[cat_col], average = avg_type)
        #acc_gt_detections = f1_score(c[cat_col], gtd[cat_col], average = avg_type)

        accuracy_categories_dirty += acc_dirty
        accuracy_categories += acc_clean
        #accuracy_categories_dboost_detections += acc_dboost_detections
        #accuracy_categories_ed2_detections += acc_ed2_detections
        #accuracy_categories_gt_detections += acc_gt_detections

        print(f'{cat_col}', acc_dirty, " vs ", acc_clean)

    accuracy_categories_dirty =  accuracy_categories_dirty / n_cat_cols
    accuracy_categories =  accuracy_categories / n_cat_cols
    accuracy_categories_dboost_detections =  accuracy_categories_dboost_detections / n_cat_cols
    accuracy_categories_ed2_detections =  accuracy_categories_ed2_detections / n_cat_cols
    accuracy_categories_gt_detections =  accuracy_categories_gt_detections / n_cat_cols

    print(f'TOTAL: {accuracy_categories_dirty} vs  {accuracy_categories} vs ED2 {accuracy_categories_ed2_detections} vs DBOOST {accuracy_categories_dboost_detections} vs GT {accuracy_categories_gt_detections}')


dirty_data[filtered_header] = df_cleaned[filtered_header]#clean_csv
lop_data = reverse_to_input_domain(args.dataset, dirty_data, FULL_SCALER, CAT_ENCODER)

# #Rebuild the whole dataset after cleaning.
# norm = (MAX - MIN) + 1e-10
# for i, clean in enumerate(clean_csv):
#     #reverse normalization
#     aux = (clean * norm)
#     clean_csv[i,:] = tf.where(MIN < 0, MIN - aux, MIN + aux)
#     #clean_csv[i,:] = denormalize_by_min_max(clean, MIN, MAX, full_header)

lop_data.to_csv(f'./DATASETS_REIN/{args.dataset}/LOP.csv', index = False)




rmse_dirty, rmse_repaired = cln.evaluate(f"./DATASETS_REIN/{args.dataset}/clean.csv",
                                         f"./DATASETS_REIN/{args.dataset}/dirty01.csv",
                                         f"./DATASETS_REIN/{args.dataset}/LOP.csv",
                                         args.eval_tuples)

print(rmse_dirty, rmse_repaired)

#save results for plots
for_plots = pd.DataFrame({'rmse_numeric':rmse_repaired, 'rmse_gt_detector':rmse_gt, 'rmse_dirty': rmse_dirty,
                          'rmse_dboost': rmse_dboost, 'rmse_ed2': rmse_ed2,
                          'accuracy_categorical': accuracy_categories,
                          'accuracy_categorical_dirty': accuracy_categories_dirty,
                          'accuracy_categorical_ed2_detections': accuracy_categories_ed2_detections,
                          'accuracy_categorical_dboost_detections': accuracy_categories_dboost_detections,
                          'accuracy_categorical_gt_detections': accuracy_categories_gt_detections}, index = [0])
for_plots.index.name = "rein"
for_plots.to_csv(f'./evaluation/rein_rmse_{args.dataset}.csv')




#######################################################################




########### WRITE DETECTION DICTIONARY ##########

#get row and column where there is no error (i.e. k = 0 = identity or equal to K)
detection_dictionary_indexes = tf.where(tf.math.logical_or(tf.equal(Ks_csv, 0), tf.equal(Ks_csv, K)))
det_idx = detection_dictionary_indexes.numpy()


#ORDER By COLUMN INDEX AND REPLACE IT WITH CORRECT ID OF THE FULL DATASET
replacement_indexes = [full_header.index(i) for i in filtered_header if i in  full_header]
#print(full_header, filtered_header, replacement_indexes)

det_idx = pd.DataFrame(det_idx).sort_values([1, 0])

for idx, rplc in enumerate(replacement_indexes):
    det_idx = det_idx.replace(str(idx), rplc)


#FIX FOR REIN BENCHMARK INDEXES ROWS JUMP HEADER AND START FROM 1, COLUMNS START FROM 0
det_idx = det_idx.apply(pd.to_numeric) + [2, 0]

#ADD "JUST A DUMMY VALUE"
dummy_column =  np.full((det_idx.shape[0], 1), "JUST A DUMMY VALUE")
det_idx = pd.DataFrame(np.concatenate([det_idx, dummy_column], axis=1))


det_idx.to_csv(f'./DATASETS_REIN/{args.dataset}/detections.csv', header = False, index = False)










# #PLOT=====================================================================
# #save the results .csv to enable third party plotting.

# df_for_plots = pd.DataFrame({'clean':clean_scores, 'dirty':n_scores, 'lop':en_smart_scores})
# df_for_plots.index.name = "percentage_dirty"
# df_for_plots.to_csv(f'./evaluation/{args.experiment}_{args.dataset}.csv')

# df = pd.DataFrame(np.column_stack((n_scores, en_scores, en_smart_scores)), columns=["Default", "LOP", "LOP + Smart"])
# ax = df.plot(kind = 'bar')#, hatch = ["."])

# labels = [item.get_text() for item in ax.get_xticklabels()]

# aux = -1
# for p in reversed(percentages_of_noise):
#     labels[aux] = str(int(p*100)) + "%"
#     aux -= 1

# plt.axhline(y=df["Default"][0], color='k', linestyle='-')
# ax.set_xticklabels(labels)
# ax.set_ylabel('RMSE')
# ax.set_ylim(bottom=max(np.amin(df["Default"]) - 1,0))

# plt.tight_layout()
# plt.autoscale()

# plt.show()
# plt.close()
