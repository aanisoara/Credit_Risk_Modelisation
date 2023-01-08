
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#--------------------------------------------------


def describe_dataset(source_files):

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data))
        files_nb_columns.append(len(file_data.columns))

    # Create a dataframe 
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns})

    presentation_df.index += 1

    return presentation_df


#------------------------------------------

def missing_values_percent_per(data):
    
    missing_percent_df = pd.DataFrame({'Percent Missing Values':data.isnull().sum()/len(data)*100}).sort_values(by="Percent Missing Values", ascending=False)
    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing Values']
    missing_percent_df['Total'] = 100

    return missing_percent_df

#------------------------------------------

def plot_percentage_missing_values_for(data, long, larg):

    data_to_plot = missing_values_percent_per(data)\
                     .sort_values("Percent Filled").reset_index()

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot Total -values
    handle_plot_1 = sns.barplot(x="Total", y="index",
                                data=data_to_plot,
                                label="non renseignées",
                                color="thistle", alpha=0.3)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(),
                                  size=TICK_SIZE)
    _, ylabels = plt.yticks()
    handle_plot_1.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot Percent Filled values
    handle_plot_2 = sns.barplot(x="Percent Filled",
                                y="index",
                                data=data_to_plot,
                                label="renseignées",
                                color="darkviolet")

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(),
                                  size=TICK_SIZE)
    handle_plot_2.set_yticklabels(ylabels, size=TICK_SIZE)


    # Ajouter la legende 
    axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="Colonnes", xlabel="Pourcentage de valeurs (%)")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE,
                    labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE,
                    labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x)) + '%'))
    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    plt.show()

#---------------------------------------------

def plot_repartition(data, title, long, larg):

    TITLE_SIZE = 20
    TITLE_PAD = 30

    f, ax = plt.subplots(figsize=(long, larg))

    # Figure title
    plt.title(title, fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
       
    # Bold text
    plt.rcParams["font.weight"] = "bold"

    # Create pie chart 
    a = data.value_counts(normalize=True).plot(kind='pie', 
                                               autopct=lambda x:'{:1.2f}'.format(x) + '%', 
                                               fontsize =15)
    # Supprime y axis label
    ax.set_ylabel('')
    
    plt.axis('equal') 
    
    plt.show()

#-------------------------------------------

def split_variables_df(df):
    dicho = []
    cat = []
    continu = []
    regrouper = []
    ordinale = []
    reste = []
    for i in range(df.shape[0]):
        if df['Colonne1'][i] == 'dicho':
            #print(df_dico['variable'][i])
            dicho.append(df['variable'][i])
        elif df['Colonne1'][i] == 'cat':
            cat.append(df['variable'][i])
        elif df['Colonne1'][i] == 'continu':
            continu.append(df['variable'][i])
        elif df['Colonne1'][i] == 'regrouper':
            regrouper.append(df['variable'][i])
        elif df['Colonne1'][i] == 'ordinale':
            ordinale.append(df['variable'][i])
        else:
            reste.append(df['variable'][i])


#-------------------------------------------

def valeur_nan(data):
    nbre_nan = data.isnull().sum()
    percent_1 = data.isnull().sum()/data.isnull().count()*100
    percent_2 = (np.round(percent_1, 2))
    missing_data = pd.concat([nbre_nan, percent_2], 
                             axis=1, keys=['Total', '%']).sort_values('%', ascending=False)
    return missing_data

#-------------------------------------------

def plotBoxPlots(data, long, larg, nb_rows, nb_cols):

    TITLE_SIZE = 25
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 10

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("VALEURS QUANTITATIVES - DISTRIBUTION", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_quant in data.columns.tolist():
        ax = axes[row, column]

        sns.despine(left=True)

        b = sns.boxplot(x=data[ind_quant], ax=ax, color="darkviolet")

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        if ind_quant == "salt_100g":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        ly = ax.get_ylabel()
        ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.tick_params(axis='both', which='major', pad=TICK_PAD)

        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0


#--------------------------------------------

def analyse_category(df):
    for var in df:
        if df[var].dtype == object:
            print(var)
            print('Nombres de catégories :', df[var].nunique())
            print('Liste des catégories : ',
                  sorted([str(e) for e in
                          df[var].unique().tolist()]))
            print('\n')

#-------------------------------------------

# Variables continues
def analyse_quanti(df):
    for var in df:
        if df[var].dtype == 'float' :
            print(var)
            print('Nombres de valeurs manquantes :',df[var].isnull().sum())
            print(df[var].describe())
            print(df[var].value_counts())


#-------------------------------------------
def nettoyage_age(x):
    if x <= 0:
        return np.nan
    else:
        return x

#-------------------------------------------

def visualisation_modalite(data, variable) : 
    
    ax, fig = plt.subplots(figsize=(20, 8)) 
    ax = sns.countplot(y = variable, data = data, order = data[variable].value_counts(ascending = False).index)

    for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_width()/len(data[variable]))
                x = p.get_x() + p.get_width()
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y), fontsize=20, fontweight='bold')

    plt.show()

#-------------------------------------------

def impact_defaut(data, variable) : 
    
    modalite = data[[variable, 'defaut_36mois']].groupby([variable], as_index = False).mean()
    modalite.sort_values(by='defaut_36mois', ascending = False, inplace = True)
    
    ax, fig = plt.subplots(figsize=(20, 12)) 
    ax = sns.barplot(x ='defaut_36mois', y = variable, data = modalite)

    for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_width())
                x = p.get_x() + p.get_width()
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y), fontsize=20, fontweight='bold')

    plt.show()

#-------------------------------------------

'''

from preprocess_fonctions import split_variables_df

def split_variables_df(df):
    dicho = []
    cat = []
    continu = []
    regrouper = []
    ordinale = []
    reste = []
    for i in range(df.shape[0]):
        if df['Colonne1'][i] == 'dicho':
            return (dicho.append(df['variable'][i]))
        elif df['Colonne1'][i] == 'cat':
            return(cat.append(df['variable'][i]))
        elif df['Colonne1'][i] == 'continu':
            return(continu.append(df['variable'][i]))
        elif df['Colonne1'][i] == 'regrouper':
            return(regrouper.append(df['variable'][i]))
        elif df['Colonne1'][i] == 'ordinale':
            return(ordinale.append(df['variable'][i]))
        else:
            return(reste.append(df['variable'][i]))


pf.split_variables_df(df_dico)
'''