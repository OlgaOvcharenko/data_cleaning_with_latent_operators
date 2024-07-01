import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from matplotlib.lines import Line2D
import numpy as np
import glob
import os
import preprocess

parser = argparse.ArgumentParser(description="Disentangled Latent Space Operator for Data Engineering")
parser.add_argument("--dataset", default='adult')
parser.add_argument("--experiment", default='vs_dirty')
parser.add_argument("--type", default='point')
parser.add_argument("--legend", action='store_true')
parser.add_argument("--y_title", action='store_true')
parser.add_argument("--x_title", action='store_true')

args = parser.parse_args()

def plot_basic(dataset, experiment, plot_type = 'point'):

    df = pd.read_csv(f'./evaluation/{experiment}_{dataset}.csv')
    df['percentage_dirty'] = df['percentage_dirty'] * 100
    df['percentage_dirty'] = df['percentage_dirty'].astype('int').astype(str) + '%'
    
    
    palette = {'clean': '#00B3AD', 'dirty': '#B33F00'}
    

    #melt all columns into value
    df = df.melt(id_vars='percentage_dirty', var_name='Model')

    if plot_type == 'point':
        g = sns.catplot(data = df, x = 'percentage_dirty', y = 'value', hue = 'Model', errorbar = None, marker=['D','o'], kind = args.type, palette=palette, linewidth=1.5)
        plt.grid()
    else:
       g = sns.catplot(data = df, x = 'percentage_dirty', y = 'value', hue = 'Model', errorbar = None, kind = args.type, palette=palette, linewidth=1.5)
    
    g.legend.remove()
    g.refline(y=df['value'][0], color='k')
    plt.xlabel('Percentage of Tuples with Errors')
    plt.ylabel('Downstream ML Model RMSE')
    plt.legend(frameon = True, facecolor = 'white')
    plt.autoscale()
    plt.tight_layout()        
    plt.savefig(f'./evaluation/plots/{args.dataset}_percentage.svg')
    plt.show()



def plot_all_datasets(dataset_list, experiment):
    df = pd.concat((pd.read_csv(f'./evaluation/{experiment}_{f}.csv').assign(filename=f) for f in dataset_list), ignore_index=True)
    print(df.head)
   
    df.drop(['percentage_dirty'], axis=1, inplace = True)
    
    palette = {'clean': '#FF6B1A', 'dirty': '#B33F00', 'lop': '#00B3AD'}


    cc = df["clean"]
    df.drop(['clean'], axis=1, inplace = True)
    
    #melt all columns into value
    df = df.melt(id_vars='filename', var_name='Model')

    num_datasets = int(df.shape[0]/2.0)
   

    fig, ax = plt.subplots()
    
    g = sns.barplot(data = df, x = 'filename', y = 'value', hue = 'Model', errorbar = None,  palette=palette, linewidth=1.5, ax = ax)
   

    for ix in range(num_datasets):
        #1 repeat for dirty, one for clean
        a = ax.patches[ix]
        x_start = a.get_x()
        width = a.get_width()
    
        ax.plot([x_start, x_start+width], 2*[cc.iloc[ix]], '--', c='w')

        a = ax.patches[ix + num_datasets]
        x_start = a.get_x()
        width = a.get_width()
    
        ax.plot([x_start, x_start+width], 2*[cc.iloc[ix]], '--', c='w')
    
    plt.xlabel('Dataset')
    plt.ylabel('Downstream ML Model RMSE')
    plt.legend(["Dirty", "LOP", "Clean"], handlelength=2, frameon = True, facecolor = 'gray', framealpha=0.2)
    plt.autoscale()
    plt.tight_layout()
    #plt.ylim(0.0 , 4.5)

    plt.savefig('./evaluation/plots/plot_vs_rein_dirty.svg')
    plt.show()


def _plot_ablation(df, experiment, x_variable, x_title, plot_type, palette):

    if plot_type == 'point':
        g = sns.catplot(data = df, x = x_variable, y = 'value', hue = 'Model', errorbar = None, marker=['D','o', '*'], kind = args.type, palette=palette, linewidth=1.5)
        plt.grid()
    else:
        g = sns.catplot(data = df, x = x_variable, y = 'value', hue = 'Model', errorbar = None, kind = args.type, palette=palette, linewidth=1.5)
      
    g.legend.remove()
    plt.xlabel(x_title)
    plt.ylabel('Downstream ML Model RMSE')
    plt.legend(frameon = True, facecolor = 'white')
    plt.autoscale()
    plt.tight_layout()        
    plt.savefig(f'./evaluation/ablation_studies/plots/{experiment}.svg')
    plt.show()


def _plot_ablation_rmse(df, experiment, x_variable, x_title, plot_type, palette):

    if plot_type == 'point':
       # g = sns.catplot(data = df, x = x_variable, y = 'value', hue = 'Model', errorbar = None, marker=['D','o', '*'], kind = args.type, palette=palette, linewidth=1.5)
        g = sns.lineplot(data = df, x = x_variable, y = 'value', hue = 'Model', errorbar = None, palette=palette, linewidth=1.5)
        #plt.grid()
    else:
        g = sns.catplot(data = df, x = x_variable, y = 'value', hue = 'Model', errorbar = None, kind = args.type, palette=palette, linewidth=1.5)
      
    g.legend_.remove()
    plt.xlabel(x_title)
    plt.ylabel('RMSE (lowers is better)')
    plt.autoscale()
    plt.tight_layout()        
    plt.savefig(f'./evaluation/ablation_studies/plots/rmse_{experiment}.svg')
    plt.show()

def plot_ablation_studies_rmse(dataset, plot_type = 'point'):

    df_ks = pd.read_csv(f'./evaluation/ablation_studies/rmse_ks_{args.dataset}.csv', usecols=["param", "lop_numeric"])[1:]
    df_tuples = pd.read_csv(f'./evaluation/ablation_studies/rmse_tuples_{args.dataset}.csv', usecols=["param", "lop_numeric"])
    df_latent = pd.read_csv(f'./evaluation/ablation_studies/rmse_latents_{args.dataset}.csv',usecols=["param", "lop_numeric"])
    df_epochs = pd.read_csv(f'./evaluation/ablation_studies/rmse_epochs_{args.dataset}.csv',usecols=["param", "lop_numeric"])

    
    palette = {'lop_numeric': '#00B3AD'}
    
    #melt all columns into value
    df_ks = df_ks.melt(id_vars='param', var_name='Model')
    df_tuples = df_tuples.melt(id_vars='param', var_name='Model')
    df_latent = df_latent.melt(id_vars='param', var_name='Model')
    df_epochs = df_epochs.melt(id_vars='param', var_name='Model')


    _plot_ablation_rmse(df_ks, f'{args.dataset}_ks', 'param', 'Value of K During Training', plot_type, palette)
    _plot_ablation_rmse(df_tuples, f'{args.dataset}_tuples', 'param', 'Number of Training Tuples', plot_type, palette)
    _plot_ablation_rmse(df_latent, f'{args.dataset}_latent', 'param', 'Dimensionality of the Latent Space', plot_type, palette)
    _plot_ablation_rmse(df_epochs, f'{args.dataset}_epochs', 'param', 'Number of Training Epochs', plot_type, palette)


def plot_ablation_studies(dataset, plot_type = 'point'):

    df_ks = pd.read_csv(f'./evaluation/ablation_studies/rmse_ks_{args.dataset}.csv')
    df_tuples = pd.read_csv(f'./evaluation/ablation_studies/rmse_tuples_{args.dataset}.csv')
    df_epochs = pd.read_csv(f'./evaluation/ablation_studies/rmse_epochs_{args.dataset}.csv')
    df_latents = pd.read_csv(f'./evaluation/ablation_studies/rmse_latents_{args.dataset}.csv')
    df_equal = pd.read_csv(f'./evaluation/ablation_studies/rmse_equal_{args.dataset}.csv')
    df_proportional = pd.read_csv(f'./evaluation/ablation_studies/rmse_proportional_{args.dataset}.csv')

    palette = {'clean': '#FF6B1A', 'dirty': '#B33F00', 'lop_numeric': '#00B3AD'}
    
    #melt all columns into value
    df_ks = df_ks.melt(id_vars='param', var_name='Model')
    df_tuples = df_tuples.melt(id_vars='param', var_name='Model')
    df_epochs = df_epochs.melt(id_vars='param', var_name='Model')
    df_latents = df_latents.melt(id_vars='param', var_name='Model')
    df_equal = df_equal.melt(id_vars='param', var_name='Model')
    df_proportional= df_proportional.melt(id_vars='param', var_name='Model')

    _plot_ablation(df_ks, f'{args.dataset}_ks', 'param', 'Number of Ks', plot_type, palette)
    _plot_ablation(df_proportional, f'{args.dataset}_proportional', 'param', 'Number of Ks, Latent is 10x', plot_type, palette)
    _plot_ablation(df_equal, f'{args.dataset}_equal', 'param', 'Latent == K', plot_type, palette)
    _plot_ablation(df_tuples, f'{args.dataset}_tuples', 'param', 'Number of Training Tuples', plot_type, palette)
    _plot_ablation(df_epochs, f'{args.dataset}_epochs', 'param', 'Number of Training Epochs', plot_type, palette)
    _plot_ablation(df_latents, f'{args.dataset}_latents', 'param', 'Latent Space Dimensionality', plot_type, palette)

def _plot_rein(dataset, data_type = "numeric", metric = "f1"):
    df = pd.read_csv(f'./DATASETS_REIN/rein_{dataset}_cleaning_results.csv')
    df_lop = pd.read_csv(f'./evaluation/rein_rmse_{args.dataset}.csv')

    df.drop(df.columns.difference(['detector', 'cleaner','onlyNum_rmse_repaired', 'onlyCat_f',  'onlyCat_p',  'onlyCat_r']), 1, inplace=True)
    
    #received the RAHA results in a different format from REIN benchmark
    if dataset == "adult" or dataset == "soccer_PLAYER" or dataset == "har":
        df_raha = pd.read_csv(f'./DATASETS_REIN/rein_{args.dataset}_raha.csv')
        df_raha.rename(columns={"tool_name": "cleaner"}, inplace= True)
        df_raha["detector"] = "raha"
        df_raha.drop(df_raha.columns.difference(['detector', 'cleaner','onlyNum_rmse_repaired', 'onlyCat_f',  'onlyCat_p',  'onlyCat_r']), 1, inplace=True)
        df_raha.sort_index(axis=1, inplace=True)
        df.sort_index(axis=1, inplace=True)
        df = pd.concat([df, df_raha], ignore_index=True)
        
    fig, ax = plt.subplots()

    #fix the model names
    df['cleaner'] = df['cleaner'].replace(regex=['standardImputer-'], value='SI ')
    df['cleaner'] = df['cleaner'].replace(regex=['mlImputer-'], value='ML ')
    df['cleaner'] = df['cleaner'].replace(regex=['seperate-'], value='')
    df['cleaner'] = df['cleaner'].replace(regex=['-dummy'], value='')
    df['cleaner'] = df['cleaner'].replace(regex=['separate-'], value='ML ')
    df['cleaner'] = df['cleaner'].replace(regex=['impute-'], value='')
    df['cleaner'] = df['cleaner'].replace(regex=['missForest'], value='MF')
    df['cleaner'] = df['cleaner'].replace(regex=['decisionTree'], value='DT')
    df['cleaner'] = df['cleaner'].replace(regex=['bayesianRidge'], value='BR')
    df.drop(df[df['cleaner'] == "SI delete"].index, inplace=True)


    #SAME COLORS ACROSS ALL PLOTS###########################    
    # create an array with all unique categories from the 'mode' column in all dataframes
    all_files = glob.glob("./DATASETS_REIN/rein_*_cleaning_results.csv")
    df_colors = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    detectors = df_colors['detector'].unique()

    # create a color palette for the number of values in modes
    colors = sns.color_palette('Set2', len(detectors))

    # create a dictionary of modes and colors
    detectors_palette = dict(zip(detectors, colors))
    ########################################################

    #for the legend to be sorted
    df.sort_values(by=['detector', 'cleaner'], inplace = True)
    

    if data_type == "numeric":
        g = sns.barplot(data = df, x = 'cleaner', y = 'onlyNum_rmse_repaired', hue = 'detector', errorbar = None,  linewidth=1.5, ax = ax, palette = detectors_palette)

        plt.axhline(y = df_lop['rmse_numeric'][0], color = 'k', linestyle = '--')
        #plt.axhline(y = df_lop['rmse_dboost'][0], color = 'g', linestyle = '--')
        plt.axhline(y = df_lop['rmse_dirty'][0], color = 'grey', linestyle = '--') 
        plt.xlabel('Error Repair Method')
        plt.ylabel('RMSE (lower is better)')

        legend_elements  =  [Line2D([0], [0], linestyle='--', color='k', label='LOP', markerfacecolor='k', markersize=15),
                         #Line2D([0], [0], linestyle='--', color='g', label='DBoost + LOP', markerfacecolor='g', markersize=15),
                         Line2D([0], [0], linestyle='--', color='grey', label='Dirty', markerfacecolor='g', markersize=15)]
        
        plt.ylim(0.0 , 1.6)
    
        
    elif data_type == "categorical":

        if metric == "f1":
            rein_col = 'onlyCat_f'
            lop_col = 'accuracy_categorical'
            title = "F1"
        elif metric == "precision":
            rein_col = 'onlyCat_p'
            lop_col = 'precision_categorical'
            title = "Precision"
        elif metric == "recall":
            rein_col = 'onlyCat_r'
            lop_col = 'recall_categorical'
            title = "Recall"

        g = sns.barplot(data = df, x = 'cleaner', y = rein_col, hue = 'detector', errorbar = None,  linewidth=1.5, ax = ax, palette = detectors_palette)
        plt.axhline(y = df_lop[lop_col][0], color = 'k', linestyle = '--')
        plt.xlabel('Error Repair Method')
        plt.ylabel(f'{title} Score (higher is better)')
        legend_elements  =  [Line2D([0], [0], linestyle='--', color='k', label='LOP', markerfacecolor='k', markersize=15)]
        plt.ylim(0.0 , 1.0)

    #change aspect ratio
    #g.set_box_aspect(40/len(df))
    g.set_box_aspect(0.5) #change 10 to modify the y/x axis ratio


    if args.legend:
        legend1 = plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.9), frameon= False)
        plt.gca().add_artist(legend1)

    plt.legend(loc='best', bbox_to_anchor=(1, 0.8), frameon= False, title = 'Error Detection Method:')
    plt.xticks(rotation=45, ha="right")

    if not args.legend:
        plt.legend('',frameon=False)

    plt.autoscale()
    plt.tight_layout()

    if not args.x_title:
        plt.xlabel('')
    if not args.y_title:
        plt.ylabel('')

    if data_type == "categorical":
        plt.savefig(f'./evaluation/plots/rein_comparision_{data_type}_{args.dataset}_{metric}.svg')
    else:
        plt.savefig(f'./evaluation/plots/rein_comparision_{data_type}_{args.dataset}.svg')
        
    plt.show()
    return g
    
def plot_rein_numeric(dataset):
    _plot_rein(dataset, "numeric")


def plot_rein_categorical(dataset):
    _plot_rein(dataset, "categorical", "f1")
    _plot_rein(dataset, "categorical", "precision")
    _plot_rein(dataset, "categorical", "recall")



def plot_tuple_wise(dataset, data_type = "numeric"):
    df = pd.read_csv(f'./DATASETS_REIN/rein_{dataset}_tuple_wise_cleaning_results.csv')
    df_lop = pd.read_csv(f'./evaluation/ablation_studies/rmse_tuples_{args.dataset}.csv')
    df.drop(df.columns.difference(['detector', 'train_size','onlyNum_rmse_repaired', 'onlyCat_f']), 1, inplace=True)

    #fix differences
    df_lop = df_lop.rename({'param': 'train_size', 'lop_numeric': 'onlyNum_rmse_repaired'}, axis=1)
    df = df.iloc[1:]
    df['ds'] = 'Best Baseline'
    df_lop['ds'] = 'LOP'

    #concatenate both datasets, sorting is done by the plot itself
    dss = pd.concat([df, df_lop])
    
    #reduce datapoints for the paper image
    dss = dss[dss["train_size"] != 15000.0]
    dss = dss[dss["train_size"] != 25000.0]
    dss = dss[dss["train_size"] != 35000.0]
    dss = dss[dss["train_size"] != 40000.0]

    dss.loc[dss["train_size"] == 45222.0, "train_size"] = "all tuples"
    
    fig, ax = plt.subplots()
    
    g = sns.barplot(data = dss, x = 'train_size', y = 'onlyNum_rmse_repaired', hue = 'ds', errorbar = None,  linewidth=1.5, ax = ax, palette='Spectral')#, marker=['D','o', '*'])

    plt.xlabel('Number of Training Tuples')
    plt.ylabel('RMSE (lower is better)')

    plt.ylim(0.0 , 1.6)

    plt.legend(loc='best', bbox_to_anchor=(1, 0.8), frameon= False, title = '')
    plt.xticks(rotation=45, ha="right")
    plt.autoscale()
    plt.tight_layout()
    
    plt.savefig(f'./evaluation/ablation_studies/plots/rein_comparision_tuple_wise_{args.dataset}.svg')
    plt.show()



def plot_time_vs_rmse(dataset):

    df_ks = pd.read_csv(f'./evaluation/ablation_studies/rmse_ks_{args.dataset}.csv', usecols=["param", "lop_numeric"])[1:]
    df_latent = pd.read_csv(f'./evaluation/ablation_studies/rmse_latents_{args.dataset}.csv',usecols=["param", "lop_numeric"])
    time_ks = pd.read_csv(f'./evaluation/ablation_studies/time_to_train_k_{args.dataset}.csv',usecols=["param", "sec"])
    time_latent = pd.read_csv(f'./evaluation/ablation_studies/time_to_train_latent_{args.dataset}.csv', usecols=["param", "sec"])

    #in minutes
    time_latent['sec'] = time_latent['sec'] / 60
    time_ks['sec'] = time_ks['sec'] / 60

    #plot bot hi nthe same axis (RMSE vs time)
    time_latent = pd.concat([time_latent, df_latent['lop_numeric']], axis = 1)
    time_ks = pd.concat([time_ks, df_ks['lop_numeric']], axis = 1)

    time_ks = time_ks[1:] # drop K = 1 because there is no cleaning

    palette = {'lop_numeric': '#00B3AD', 'sec': '#FFB3AD'}
    
    #melt all columns into value
    #df_ks = df_ks.melt(id_vars='param', var_name='Model')
    #df_latent = df_latent.melt(id_vars='param', var_name='Model')
    #time_ks = time_ks.melt(id_vars='param', var_name='Model')
    #time_latent = time_latent.melt(id_vars='param', var_name='Model')

    #latent plot
    fig, ax1 = plt.subplots()

    g = sns.lineplot(x = time_latent["param"],  y = time_latent["sec"],  errorbar = None, marker='o', palette=palette["sec"], linewidth=2.5, ax = ax1)

    ax2 = plt.twinx()

    g2 = sns.lineplot(x = time_latent["param"], y = time_latent["lop_numeric"],  errorbar = None, marker='o', color=palette["lop_numeric"], linewidth=2.5, ax = ax2)

    ax2.set(ylim=(0, 1))
        
    ax1.set(xlabel = 'Dimensionality of the latent space (per column)')
    ax1.set(ylabel='Time to train in minutes')
    ax2.set(ylabel='F1 score on data cleaning')
    plt.tight_layout()        
    plt.savefig(f'./evaluation/ablation_studies/plots/time_latent_{args.dataset}.svg')
    plt.show()

    #K version of the plot
    fig, ax1 = plt.subplots()
    
    g = sns.lineplot(x = time_ks["param"], y = time_ks["sec"],  errorbar = None, marker='o', palette=palette["sec"], linewidth=2.5, ax = ax1)

    ax2 = plt.twinx()

    g2 = sns.lineplot(x = time_ks["param"], y = time_ks["lop_numeric"],  errorbar = None, marker='o', color=palette["lop_numeric"], linewidth=2.5, ax = ax2)

    ax1.set(ylim=(0, 20))
    ax2.set(ylim=(0, 2))
        
    ax1.set(xlabel = 'Number of transformations (K)')
    ax1.set(ylabel='Time to train in minutes')
    ax2.set(ylabel='F1 score on data cleaning')
    plt.tight_layout()        
    plt.savefig(f'./evaluation/ablation_studies/plots/time_ks_{args.dataset}.svg')
    plt.show()



if args.experiment == "vs_dirty_percentages" : 
    plot_basic(args.dataset, args.experiment, args.type)
elif args.experiment == "vs_dirty" : 
    plot_all_datasets(["adult", "beers", "bikes", "smart_factory", "soccer_OR", "soccer_PLAYER"], args.experiment)#, args.type)
elif args.experiment == "ablation":
    plot_ablation_studies(args.dataset, args.type)
elif args.experiment == "ablation_rmse":
    plot_ablation_studies_rmse(args.dataset, args.type)
elif args.experiment == "compare_to_rein_numeric":
    plot_rein_numeric(args.dataset)
elif args.experiment == "compare_to_rein_categorical":
    plot_rein_categorical(args.dataset)
elif args.experiment == "tuplewise":
    plot_tuple_wise(args.dataset)
elif args.experiment == "performance":
    plot_time_vs_rmse(args.dataset)
