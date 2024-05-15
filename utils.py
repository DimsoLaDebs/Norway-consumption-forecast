import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from joblib import dump







def preprocessing(df, area = 4, train_set_size =0.8, test_set_days = 10, previous_y = False):

    '''
    Preprocess the data by scaling it, adding features, separating the set and selecting data of one area

    inputs :
    df (dataframe) : data initial
    area (int) : area of interest, from 1 to 5. area default set to 0. If stays at 0, there is no area selected and the output just separate target and features for the whole datasets.
    set_size (list) : list of relative size of the train, validation and test set

    outputs :
    X_train, y_train, X_val, y_val, X_test, y_test
    
    '''

    #change timestamp in datetime format 
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    #add month, day, year columns
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour

    df.set_index('timestamp', inplace=True)

    # Créer les nouvelles colonnes moyenne journalière pour chaque 'NOi_consumption'
    for i in range(1, 6):

        daily_avg = df[f'NO{i}_consumption'].resample('D').mean()

        # Décaler la moyenne quotidienne pour l'associer à chaque heure du jour suivant
        next_day_avg = daily_avg.shift(1)  # Décale les données d'un jour

        # Étendre la moyenne décalée pour qu'elle corresponde à chaque heure du jour j+1
        # Il faut d'abord reindexer pour s'assurer que nous avons une entrée pour chaque heure
        hourly_next_day_avg = next_day_avg.reindex(df.index, method='ffill')

        # Ajouter la nouvelle caractéristique au DataFrame original
        df[f'NO{i}_consumption_avg_prev_day'] = hourly_next_day_avg

        # Attribuer aux 24 premières entrées la 25e valeur
        df.loc[df.index[:24], f'NO{i}_consumption_avg_prev_day'] = df.loc[df.index[24], f'NO{i}_consumption_avg_prev_day']



    #set separation
    n = len(df)
    df_train = df[:int(n * train_set_size)]
    df_val = df[int(n * train_set_size): -test_set_days*24]
    df_test = df[-10*24:]



    # Scaling data
    scaler = StandardScaler()
    features = ['NO1_consumption', 'NO1_temperature', 'NO2_consumption', 'NO2_temperature', 'NO3_consumption', 'NO3_temperature', 'NO4_consumption', 'NO4_temperature', 'NO5_consumption', 
            'NO5_temperature', 'month', 'day', 'hour', 'NO1_consumption_avg_prev_day', 'NO2_consumption_avg_prev_day', 'NO3_consumption_avg_prev_day', 'NO4_consumption_avg_prev_day', 'NO5_consumption_avg_prev_day']
    scaler.fit(df_train[features])
    # Transformer les données
    standardized_trainset = scaler.transform(df_train[features]) #numpy array
    standardized_valset = scaler.transform(df_val[features]) #numpy array
    standardized_testset = scaler.transform(df_test[features]) #numpy array
    # Convertir en DataFrame pour une utilisation ultérieure
    df_train_scaled = pd.DataFrame(standardized_trainset, columns=features)
    df_val_scaled = pd.DataFrame(standardized_valset, columns=features)
    df_test_scaled = pd.DataFrame(standardized_testset, columns=features)

    dump(scaler, 'scaler.joblib')

    # Creation of new 'NOi_consumption' shifted columns
    for i in range(1, 6):
        col_name = f'NO{i}_consumption'
        if previous_y:
            df_train_scaled[f'{col_name}_1h_before'] = df_train_scaled[col_name].shift(1)
        df_train_scaled[f'{col_name}_24h_before'] = df_train_scaled[col_name].shift(24)
        
    # Pour df_val_scaled
    for i in range(1, 6):
        col_name = f'NO{i}_consumption'
        if previous_y:
            df_val_scaled[f'{col_name}_1h_before'] = df_val_scaled[col_name].shift(1)
        df_val_scaled[f'{col_name}_24h_before'] = df_val_scaled[col_name].shift(24)
        
    # Pour df_test_scaled
    for i in range(1, 6):
        col_name = f'NO{i}_consumption'
        if previous_y:
            df_test_scaled[f'{col_name}_1h_before'] = df_test_scaled[col_name].shift(1)
        df_test_scaled[f'{col_name}_24h_before'] = df_test_scaled[col_name].shift(24)
        


    # Replace NaN in new columns
    for i in range(1, 6):
        col_name = f'NO{i}_consumption'
        if previous_y:
            col_1h_before = f'{col_name}_1h_before'
            df_train_scaled[col_1h_before].fillna(df_train_scaled[col_name], inplace=True)


        col_24h_before = f'{col_name}_24h_before'
        # Remplacer les NaN dans les colonnes 24h_before avec les valeurs actuelles de consommation
        df_train_scaled[col_24h_before].fillna(df_train_scaled[col_name], inplace=True)
        
        
    # Pour df_val_scaled
    for i in range(1, 6):
        col_name = f'NO{i}_consumption'
        if previous_y:
            col_1h_before = f'{col_name}_1h_before'
            df_val_scaled[col_1h_before].fillna(df_val_scaled[col_name], inplace=True)

        col_24h_before = f'{col_name}_24h_before'
        df_val_scaled[col_24h_before].fillna(df_val_scaled[col_name], inplace=True)
        

    # Pour df_test_scaled
    for i in range(1, 6):
        col_name = f'NO{i}_consumption'
        if previous_y:
            col_1h_before = f'{col_name}_1h_before'
            df_test_scaled[col_1h_before].fillna(df_test_scaled[col_name], inplace=True)

        col_24h_before = f'{col_name}_24h_before'
        df_test_scaled[col_24h_before].fillna(df_test_scaled[col_name], inplace=True)
        

    if previous_y:

        df_train_scaled = df_train_scaled[[f'NO{area}_consumption', f'NO{area}_temperature', f'NO{area}_consumption_1h_before', f'NO{area}_consumption_24h_before', f'NO{area}_consumption_avg_prev_day', 'month', 'day', 'hour']]
        df_val_scaled = df_val_scaled[[f'NO{area}_consumption', f'NO{area}_temperature', f'NO{area}_consumption_1h_before', f'NO{area}_consumption_24h_before', f'NO{area}_consumption_avg_prev_day', 'month', 'day', 'hour']]
        df_test_scaled = df_test_scaled[[f'NO{area}_consumption', f'NO{area}_temperature', f'NO{area}_consumption_1h_before', f'NO{area}_consumption_24h_before', f'NO{area}_consumption_avg_prev_day', 'month', 'day', 'hour']]
        
        X_train = df_train_scaled.drop(f'NO{area}_consumption', axis=1)  
        y_train = df_train_scaled[f'NO{area}_consumption']

        X_val = df_val_scaled.drop(f'NO{area}_consumption', axis=1)  
        y_val = df_val_scaled[f'NO{area}_consumption']

        X_test = df_test_scaled.drop(f'NO{area}_consumption', axis=1)  
        y_test = df_test_scaled[f'NO{area}_consumption']

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    else:
        df_train_scaled = df_train_scaled[[f'NO{area}_consumption', f'NO{area}_temperature', f'NO{area}_consumption_24h_before', f'NO{area}_consumption_avg_prev_day', 'month', 'day', 'hour']]
        df_val_scaled = df_val_scaled[[f'NO{area}_consumption', f'NO{area}_temperature', f'NO{area}_consumption_24h_before', f'NO{area}_consumption_avg_prev_day', 'month', 'day', 'hour']]
        df_test_scaled = df_test_scaled[[f'NO{area}_consumption', f'NO{area}_temperature', f'NO{area}_consumption_24h_before', f'NO{area}_consumption_avg_prev_day', 'month', 'day', 'hour']]
        
        X_train = df_train_scaled.drop(f'NO{area}_consumption', axis=1)  
        y_train = df_train_scaled[f'NO{area}_consumption']

        X_val = df_val_scaled.drop(f'NO{area}_consumption', axis=1)  
        y_val = df_val_scaled[f'NO{area}_consumption']

        X_test = df_test_scaled.drop(f'NO{area}_consumption', axis=1)  
        y_test = df_test_scaled[f'NO{area}_consumption']

        return X_train, y_train, X_val, y_val, X_test, y_test












