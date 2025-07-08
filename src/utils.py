def load_data(data_path:str):
    """
    Loads the CSV data into a dataframe object from given file path and drops 'Patient Id' column from it. 
    Then returns the cleaned dataframe object.
    """
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_data_path = os.path.join(cwd, Path(data_path))

    if os.path.exists(full_data_path) and full_data_path.endswith('.csv'):
        import pandas as pd

        df = pd.read_csv(full_data_path, index_col=0)
        df.drop(['Patient Id', 'Gender'], axis=1, inplace=True)

        return df
    
    else:
        raise ValueError('Invalid path or file extension!')
    
#-----------------------------------------------------------------------------------------------------------------

def preprocess_data(df, col_transformer_path:str='models/col_transformer_fitted.joblib', 
                    target_encoder_path:str='models/ordinal_encoder_fitted.joblib', 
                    best_features_path=None):
    
    x_train, x_test, y_train, y_test = split_data(df)
    
    col_transformer = load_model(col_transformer_path)
    x_train_new = col_transformer.transform(x_train)
    x_test_new = col_transformer.transform(x_test)

    target_encoder = load_model(target_encoder_path)
    y_train_new = target_encoder.transform(y_train.values.reshape(-1,1))
    y_test_new = target_encoder.transform(y_test.values.reshape(-1,1))

    y_train_new = y_train_new.rename(columns={'x0':'Level'})
    y_test_new = y_test_new.rename(columns={'x0':'Level'})

    return x_train_new, x_test_new, y_train_new, y_test_new
    # best_features = get_best_features(best_features_path)
    # return x_train_new[best_features], x_test_new[best_features], y_train_new, y_test_new



#--------------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_model_path = os.path.join(cwd, Path(model_path))

    if os.path.exists(full_model_path) and full_model_path.endswith('.joblib'):
        from joblib import load

        with open(full_model_path, 'rb') as f:
            model = load(f)
            return model
    
    else:
        raise ValueError('Invalid path or file extension!')

#---------------------------------------------------------------------------------------------------------------

def save_model(model_object, model_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_model_path = os.path.join(cwd, Path(model_path))

    if os.path.exists(full_model_path) and full_model_path.endswith('.joblib'):
        from joblib import dump
        with open(full_model_path, 'wb') as f:
            dump(model_object, f)
            print('Saved model successfully!')

    else:
        raise ValueError('Invalid model path or file extension')

#-------------------------------------------------------------------------------------------------------------

def split_data(df):
    from sklearn.model_selection import train_test_split

    X = df.drop(['Level'], axis=1)
    Y = df['Level']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=42)
    
    return x_train, x_test, y_train, y_test

#---------------------------------------------------------------------------------------------------------------------

def get_best_features(best_features_path:str=''):
    
    best_features = load_model(best_features_path)
    return best_features

#---------------------------------------------------------------------------------------------------------------

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)

    return model

#-----------------------------------------------------------------------------------------------------------------

def test_model(model, x_test, y_test):
    score = model.score(x_test, y_test)

    return score

#-----------------------------------------------------------------------------------------------------------------

def get_value_range(df) -> dict:
    from collections import defaultdict

    features_range = defaultdict(tuple)
    for col in df.columns:
        features_range[col] = (min(df[col]), max(df[col]))

    return features_range

#---------------------------------------------------------------------------------------------------------------------

def get_test_data():
    import random
    import pandas as pd

    feature_values = {'Age': [random.choice(range(14, 73))], 
                      'Air Pollution': [random.choice(range(1, 8))], 
                      'Alcohol use': [random.choice(range(1, 8))], 
                      'Dust Allergy': [random.choice(range(1, 8))],
                      'OccuPational Hazards': [random.choice(range(1, 8))], 
                      'Genetic Risk': [random.choice(range(1, 7))], 
                      'chronic Lung Disease': [random.choice(range(1, 7))], 
                      'Balanced Diet': [random.choice(range(1, 7))], 
                      'Obesity': [random.choice(range(1, 7))], 
                      'Smoking': [random.choice(range(1, 8))], 
                      'Passive Smoker': [random.choice(range(1, 8))], 
                      'Chest Pain': [random.choice(range(1, 9))], 
                      'Coughing of Blood': [random.choice(range(1, 9))], 
                      'Fatigue': [random.choice(range(1, 9))], 
                      'Weight Loss': [random.choice(range(1, 8))], 
                      'Shortness of Breath': [random.choice(range(1, 9))], 
                      'Wheezing': [random.choice(range(1, 8))], 
                      'Swallowing Difficulty': [random.choice(range(1, 8))], 
                      'Clubbing of Finger Nails': [random.choice(range(1, 9))], 
                      'Frequent Cold': [random.choice(range(1, 7))],
                      'Dry Cough': [random.choice(range(1, 7))], 
                      'Snoring': [random.choice(range(1, 7))]}
    
    return pd.DataFrame(feature_values)

#---------------------------------------------------------------------------------------------------------------

