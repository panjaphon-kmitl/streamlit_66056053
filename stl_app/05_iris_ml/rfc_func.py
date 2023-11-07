import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def random_forrest_classifier(df:pd.DataFrame, x:list, y:str):
    '''
    A custom function that builds an RFC model.
    Since both files use this function, it would be more maintainable to put it in a .py file to import.

    Parameters:
        df (DataFrame): The whole dataframe
        x (list): A list of X columns
        y (str): Y column name

    Returns:
        model, uniques of Y, model score
    '''
    df = df.dropna()
    output = df[y]
    output, uniques = pd.factorize(output)

    features = df[x]
    features = pd.get_dummies(features)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        output,
        test_size=0.8
    )
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)

    score = accuracy_score(y_pred, y_test)

    return rfc, uniques, score