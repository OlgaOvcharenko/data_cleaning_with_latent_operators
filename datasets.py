import os
import tensorflow as tf
import numpy as np
import pandas as pd
#import diffml_data.preprocess as pr
import preprocess as pr
from sklearn import preprocessing
from copy import deepcopy
from sklearn.preprocessing import StandardScaler





def _to_tf(X,Y, batchsize):
    _dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    _dataset = _dataset.cache().shuffle(100000, reshuffle_each_iteration=True).batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    #_dataset = _dataset.cache().shuffle(100000, reshuffle_each_iteration=True).batch(batchsize).prefetch(tf.data.AUTOTUNE)
    return _dataset


def get_tf_database(data, target, batchsize):
    return _to_tf(data, target , batchsize)



def load_mnist(n_train_instances, n_test_instances):
    from tensorflow.keras.datasets import mnist    

    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    D = 784
    
    y_train = y_train[0:n_train_instances]
    y_test = y_test[0:n_test_instances]

    x_train = (x_train/255.0).reshape(-1,D)[0:n_train_instances,:]
    x_test = (x_test/255.0).reshape(-1,D)[0:n_test_instances,:]

    x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)

    return x_train, y_train, x_test, y_test


def load_california(n_train_instances, n_test_instances):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    X,y = fetch_california_housing(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    x_train = x_train[0:n_train_instances,:]
    y_train = y_train[0:n_train_instances]
    x_test = x_test[0:n_test_instances,:]
    y_test = y_test[0:n_test_instances]


    #normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)

    x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test,dtype=tf.float32)


    return x_train, y_train, x_test, y_test



##########################################################################################################









def _get_data_config():
    
    return {

#        "bikes": {"id_cols": ["instant"], "cat_cols": [], "date_cols": [("dteday", "%m/%d/%Y")], "target": "cnt"},
        "bikes": {"id_cols": ["instant"], "cat_cols": [], "date_cols": ["dteday"], "target": "cnt"}, #REIN version date is already ok
        "smart_factory": {"id_cols": [], "cat_cols": [], "date_cols":[], "target": "labels"},#["o_w_blo_voltage", "o_w_bhl_voltage","o_w_bhr_voltage","o_w_bru_voltage"], "date_cols":[], "target": "labels"},
        "water": {"id_cols": ["index"], "cat_cols": [], "date_cols":[], "target": "x38"}, #RANDOM TARGET BECAUSE IT IS A clustering TASK
        "adult": {"id_cols": [], "cat_cols": ["workclass", "education", "marital_status", "occupation", "relationship", "gender", "native_country","race"], "date_cols":[], "target": "income"},
        "beers": {"id_cols": ["index", "id", "brewery_id", "beer_name"], "cat_cols": ["city",  "state", "brewery_name", "style"], "date_cols": [], "target": "style"},
        "breast_cancer": {"id_cols":  ["Sample code number"],"cat_cols": [], "date_cols": [], "target": "class"},
        "soilmoisture": {"id_cols": [], "cat_cols": [], "date_cols": [("datetime", None)], "target": "soil_moisture"},
        "airbnb": {"id_cols": [], "cat_cols": ["LocationName", "Rating"], "date_cols": [], "target": "Price"},
        "soccer_OR": {"id_cols": ['id_x'], "cat_cols": ["player_name", "attacking_work_rate", "preferred_foot"], "date_cols": ["date", "birthday"], "target": "overall_rating"},
        "soccer_PLAYER": {"id_cols": ['id_x'], "cat_cols": ["player_name", "attacking_work_rate", "preferred_foot"], "date_cols": ["date", "birthday"], "target": "player_name"},
        "nasa": {"id_cols": [], "cat_cols": [], "date_cols": [], "target": "sound_pressure_level"}
    }


def split_numerical_and_categorical_columns(ds, filtered_header):
    dataset_config = _get_data_config()
    
    numeric_header = [i for i in filtered_header if i not in dataset_config[ds]["cat_cols"]]
    # must do the loop to keep the order
    categorical_header = [i for i in filtered_header if i in dataset_config[ds]["cat_cols"]]
    
    return numeric_header, categorical_header


def _normalize_by_min_max(df, min_x, max_x):
    
    norm = (max_x - min_x) + 1e-10 #only breaks if MAX = MIN, so we need some noise
    print(df.columns)

    normalized_data = pd.DataFrame(tf.where(min_x < 0, (df + min_x) / norm, (df - min_x) / norm).numpy(), columns = df.columns)

    #print("HERE:",df.head(), b.head())
    
    return  normalized_data


def _normalize_by_mean(df, scaler):
    return  pd.DataFrame(scaler.transform(df), columns = df.columns)

def _fit_mean_scaler(df, scaler):
    normalized_data = scaler.fit_transform(df)
    return  pd.DataFrame(normalized_data, columns = df.columns), scaler


# def denormalize_by_min_max(df, min_x, max_x, cols):

#     norm = (max_x - min_x) + 1e-10
#     aux = df * norm


#     print(df[0:10],cols)
    
#     normalized_data = pd.DataFrame(tf.where(min_x < 0, aux - min_x, aux + min_x).numpy(), columns = cols)

#     #print("HERE:",df.head(), b.head())
    
#     return  normalized_data


def _get_train_val_and_test(df, target, n_train_instances, n_test_instances):

    train, val, test = np.split(df.sample(frac=1), #shuffle
                                [int(.70*len(df)), int(.80*len(df))])
    #train, val, test = np.split(df.sample(frac=1), #shuffle
    #                            [int(.99*len(df)), int(.001*len(df))])


    X_train = train.to_numpy(dtype=np.float32)[:n_train_instances, df.columns != target ]
    X_val = val.to_numpy(dtype=np.float32)[:, df.columns != target]
    X_test = test.to_numpy(dtype=np.float32)[:n_test_instances, df.columns != target]

    #must add a newaxis (empty)
    y_train = train.to_numpy(dtype=np.float32)[:n_train_instances, df.columns == target, np.newaxis]
    y_val = val.to_numpy(dtype=np.float32)[:,df.columns == target, np.newaxis]
    y_test = test.to_numpy(dtype=np.float32)[:n_test_instances,df.columns == target, np.newaxis]

    return  X_train, y_train, X_test, y_test, X_val, y_val
    




def _replace_nan(df, missing):
    df = df.replace('N/A', missing)
    df = df.replace('', missing)
    df = df.replace(' ', missing)
    df = df.replace('NaN', missing)
    df = df.fillna(missing)
    return df





def load_regression(ds, n_train_instances, n_test_instances, normalize_y = False, normalize_sklearn = True):
    scaler = StandardScaler()
    dataset_config = _get_data_config()
    clean_dir = f"DATASETS_REIN/{ds}/"
    df = pd.read_csv(os.path.join(clean_dir, 'clean.csv'))

   
    df, CAT_ENCODER = pr.preprocess(df,dataset_config[ds]["target"],
                               dataset_config[ds]["date_cols"],
                               dataset_config[ds]["id_cols"],
                               dataset_config[ds]["cat_cols"])


    print("Unique Values per column:\n ", df.nunique())

    #drop rows that contains empty cells
    df.dropna(inplace = True)

    MIN = np.min(df,axis=0)
    MAX = np.max(df,axis=0)

    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()

    if normalize_sklearn:
        df, scaler = _fit_mean_scaler(df, scaler)
    else:
        df = _normalize_by_min_max(df, MIN, MAX)

    # when all values of the column are equal it divides by zero and return NaN, so we need to replace by 0.0!!!
    df = df.fillna(0.0)
    
    if not normalize_y:
        df[target] = y_
    
    X_train, y_train, X_test, y_test, X_val, y_val = _get_train_val_and_test(df, target,  n_train_instances, n_test_instances)
    
    return X_train, y_train, X_test, y_test, MAX, MIN, scaler, CAT_ENCODER









def load_regression_dirty(ds, n_train_instances, n_test_instances, missing, scaler, CAT_ENCODER, normalize_y = False, MAX = None, MIN = None, normalize_sklearn = True):
    dataset_config = _get_data_config()
    clean_dir = f"DATASETS_REIN/{ds}/"
    df = pd.read_csv(os.path.join(clean_dir, 'dirty01.csv'))

    df = _replace_nan(df, missing)

    #drop rows that still contains empty cells
    #df.dropna(inplace = True)
    df,_ = pr.preprocess(df,dataset_config[ds]["target"],
                         dataset_config[ds]["date_cols"],
                         dataset_config[ds]["id_cols"],
                         dataset_config[ds]["cat_cols"],
                         les = CAT_ENCODER)

    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()
    
    if normalize_sklearn:
        df = _normalize_by_mean(df, scaler)
    elif MAX is not None:
        df = _normalize_by_min_max(df, MIN, MAX)
    else:
        #df = (df - np.min(df,axis=0)) / ((np.max(df,axis=0) - np.min(df,axis=0)) + 1e-10)
        df = _normalize_by_min_max(df, np.min(df,axis=0), np.max(df,axis=0))

    # For min/max normalization, when all values of the column are equal it divides by zero and return NaN,
    #so we need to replace by 0.0!!!
    df = df.fillna(0.0) 
        
    # #TODO: HOW TO DEAL WITH ALL THIS NORMALIZATION??==========
    # df.where(df >= -float(missing), float(missing), inplace=True)
    # df.where(df <= float(missing), float(missing), inplace=True)


    print(np.min(df,axis=0), "\n",  np.max(df,axis=0))

    if not normalize_y:
        df[target] = y_    

    X_train, y_train, X_test, y_test, X_val, y_val = _get_train_val_and_test(df, target,  n_train_instances, n_test_instances)

    return X_train, y_train, X_test, y_test







def load_features_and_data(ds, n_instances, n_test_instances, missing, scaler, CAT_ENCODER,  normalize_y = False, MAX = None, MIN = None, normalize_sklearn = True):
    
    dataset_config = _get_data_config()

    clean_dir = f"DATASETS_REIN/{ds}/"
    df = pd.read_csv(os.path.join(clean_dir, 'dirty01.csv')) [0:n_instances]
    df_clean = pd.read_csv(os.path.join(clean_dir, 'clean.csv')) [0:n_instances]

    numeric_header = df_clean.select_dtypes(include="number").columns
    categorical_header = df_clean.select_dtypes(exclude="number").columns

    #remove date columns fro mthe categorical set
    categorical_header = categorical_header.drop(dataset_config[ds]["date_cols"])

    
    #df = _replace_nan(df, "0.0")
    #df_clean = _replace_nan(df_clean, "0.0")

    #save the full version for CSV write
    FULL = deepcopy(df)    
  
    #KEEP ID COLUMNS CLEAN <<<<< NOT FOR REIN???
    #df[dataset_config[ds]["date_cols"]] = df_clean[dataset_config[ds]["date_cols"]]
    #df[dataset_config[ds]["id_cols"]] = df_clean[dataset_config[ds]["id_cols"]]                   
 
    df, _ = pr.preprocess(df,
                          dataset_config[ds]["target"],
                          dataset_config[ds]["date_cols"],
                          dataset_config[ds]["id_cols"],
                          dataset_config[ds]["cat_cols"],
                          les = CAT_ENCODER)

    df_clean, _ = pr.preprocess(df_clean,
                                dataset_config[ds]["target"],
                                dataset_config[ds]["date_cols"],
                                dataset_config[ds]["id_cols"],
                                dataset_config[ds]["cat_cols"],
                                les = CAT_ENCODER,
                                drop_columns = False)

    # #FIT THE SCALER FOR ALL COLUMNS##############################
    sc = StandardScaler()
    _, full_scaler = _fit_mean_scaler(df_clean, sc)
    # #################################################################


    FULL, _ = pr.preprocess(FULL,
                            dataset_config[ds]["target"],
                            dataset_config[ds]["date_cols"],
                            dataset_config[ds]["id_cols"],
                            dataset_config[ds]["cat_cols"],
                            les = CAT_ENCODER,
                            drop_columns = False)


    #get index of unknown categories to replace for missing values later.
    #indexes_unknown_category = FULL.where(FULL == -1).stack().index.to_list() 
    #print(indexes_unknown_category)
    #print(FULL[indexes_unknown_category])

    
    header = df.columns.tolist()
    full_header = df_clean.columns.tolist()


    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()

    #Normalize    
    if normalize_sklearn:
        df = _normalize_by_mean(df, scaler)
        df_clean = _normalize_by_mean(df_clean, full_scaler)
        FULL = _normalize_by_mean(FULL, full_scaler)
    # elif MAX is not None:
    #     df = _normalize_by_min_max(df, MIN, MAX)
    #     df_clean = _normalize_by_min_max(df_clean, np.min(df_clean,axis=0), np.max(df_clean,axis=0))
    #     FULL = _normalize_by_min_max(FULL, np.min(FULL,axis=0), np.max(FULL,axis=0))
    # else:
    #     df = _normalize_by_min_max(df, np.min(df,axis=0), np.max(df,axis=0))
    #     df_clean = _normalize_by_min_max(df_clean, np.min(df_clean,axis=0), np.max(df_clean,axis=0))
    #     FULL = _normalize_by_min_max(FULL, np.min(FULL,axis=0), np.max(FULL,axis=0))


    # when all values of the column are equal it divides by zero and return NaN, so we need to replace by 0.0!!!
    #df = df.fillna(0.0)
    #FULL = FULL.fillna(0.0)
    #df_clean = df_clean.fillna(0.0)

    df_dirty = deepcopy(FULL)


    #replace NaNs as in REIN
    df_dirty = pd.DataFrame(np.nan_to_num(df_dirty), columns = df_dirty.columns)
    FULL = pd.DataFrame(np.nan_to_num(FULL), columns = FULL.columns)
    df_clean =  pd.DataFrame(np.nan_to_num(df_clean), columns = df_clean.columns)


    #0 FOR REIN###########################################
    #FULL = FULL.where(FULL >= -float(missing), 0.0)
    #FULL = FULL.where(FULL <= float(missing), 0.0)

    #MISSING VALUE FOR LOP###########################################
    df_dirty = df_dirty.where(df_dirty >= -float(missing), float(missing))
    df_dirty = df_dirty.where(df_dirty <= float(missing), float(missing))
    #only makes sense for training?

    if not normalize_y:
        df[target] = y_

    data_no_target = df.drop([target], axis = 1)
    filtered_header = data_no_target.columns.tolist()
    Y = df[target].to_numpy()#dtype=np.float32)

    #numeric_header, categorical_header = split_numerical_and_categorical_columns(ds, filtered_header)

    
    print("numeric:", numeric_header)
    print("categorical:", categorical_header)

    headers = {"full_header" : full_header ,
               "filtered_header_with_y": header,
               "filtered_header": filtered_header,
               "numeric_header": numeric_header,
               "categorical_header": categorical_header}
   
   

    return headers, target, FULL, data_no_target.to_numpy(), df_dirty, Y, df_clean, full_scaler, CAT_ENCODER


def reverse_categorical_columns(ds, data, label_encoder):
    dataset_config = _get_data_config()
    return pr.reverse_categorical_columns(ds, data, label_encoder, dataset_config)

def reverse_to_input_domain(ds, data, scaler, CAT_ENCODER):
    lop_data = pd.DataFrame(scaler.inverse_transform(data), columns = data.columns)
    return reverse_categorical_columns(ds, lop_data, CAT_ENCODER)
