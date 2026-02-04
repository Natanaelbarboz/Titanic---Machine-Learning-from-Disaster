def modelo(base):
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.model_selection import train_test_split
    from sklearn import ensemble, preprocessing, tree
    from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from yellowbrick.classifier import ConfusionMatrix, ROCAUC
    from yellowbrick.model_selection import LearningCurve

    from sklearn.experimental import enable_iterative_imputer
    from sklearn import impute
    from sklearn.ensemble import RandomForestClassifier

    import joblib 

    import os
    import pandas as pd

    from sklearn.impute import SimpleImputer

    print('Carregando arquivo csv.......')

    df = pd.read_csv(base)
    
    print('Fazendo trativas no arquivo......')

    df['Cabin_filled'] = df['Cabin'].notna().astype(int)

    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    df['Title'] = df['Title'].replace(
    ['Mlle', 'Ms'], 'Miss'
    )

    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 
        'Major', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'the Countess'],
        'Rare'
    )

    title_map = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 2,
    'Master': 3,
    'Rare': 4,
    'Dr': 5,
    'Rev': 6
    }

    df['Title_encoded'] = df['Title'].map(title_map)

    def sexo(linha):
        if linha['Sex'] == 'female':
            return 1
        else:
            return 0

    df['Sex'] = df.apply(sexo, axis=1)

    def embarqu(linha):
        if linha['Embarked'] == 'S':
            return 1
        if linha['Embarked'] == 'C':
            return 2
        if linha['Embarked'] == 'Q':
            return 3
        else:
            return 0

    df['Embarked'] = df.apply(embarqu, axis=1)

    def family(linha):
        if linha['SibSp'] > 0 or linha['Parch'] > 0:
            return 1
        else:
            return 0

    df['Family'] = df.apply(family, axis=1)
    

    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Title', 'Fare'], axis=1)

    y = df.Survived
    x = df.drop(columns=['Survived'], axis=1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

    num_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Cabin_filled', 'Title_encoded', 'Family']

    # imputer = impute.IterativeImputer()
    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(X_train[num_cols])

    X_train.loc[:,num_cols] = imputed
    imputed = imputer.transform(X_test[num_cols])
    X_test.loc[:, num_cols] = imputed


    cols = "PassengerId, Pclass, Sex, Age, SibSp, Parch, Embarked, Cabin_filled, Title_encoded, Family".split(",")

    sca = preprocessing.StandardScaler()
    X_train = sca.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = sca.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=cols)

    rf5 = ensemble.RandomForestClassifier(
    **{'max_features': 0.4, 'min_samples_leaf': 1, 'n_estimators': 200, 'random_state': 42}
    )
    rf5.fit(X_train, y_train)
    rf5.score(X_test, y_test)

    rf5.score(X_test, y_test)
    print(f'Score: {rf5.score(X_test, y_test)}')

    joblib.dump({
        'model': rf5,
        'scaler': sca,
        'imputer': imputer,
        'columns': cols,
        'title_map': title_map
    }, 'pipeline_modelo_rf1.pkl')


    print('Modelo salvo')

    mapping = {0: "died", 1: "survived"}

    fig, ax = plt.subplots(figsize=(6, 6))

    cm_viz = ConfusionMatrix(
        rf5,
        classes=["died", "survived"],
        label_encoder=mapping,
        ax=ax
    )

    cm_viz.fit(X_train, y_train)
    cm_viz.score(X_test, y_test)
    cm_viz.show()

base = '../Bases/train.csv'

modelo(base)

def carregar_modelo(path='pipeline_modelo_rf1.pkl'):
    import joblib

    pipeline = joblib.load(path)

    return (
        pipeline['model'],
        pipeline['scaler'],
        pipeline['imputer'],
        pipeline['columns'],
        pipeline['title_map']
    )

def preprocessar_novos_dados(df, title_map, columns):

    df['Cabin_filled'] = df['Cabin'].notna().astype(int)

    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    df['Title'] = df['Title'].replace(
    ['Mlle', 'Ms'], 'Miss'
    )

    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 
        'Major', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'the Countess'],
        'Rare'
    )

    title_map = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 2,
    'Master': 3,
    'Rare': 4,
    'Dr': 5,
    'Rev': 6
    }

    df['Title_encoded'] = df['Title'].map(title_map)

    def sexo(linha):
        if linha['Sex'] == 'female':
            return 1
        else:
            return 0

    df['Sex'] = df.apply(sexo, axis=1)

    def embarqu(linha):
        if linha['Embarked'] == 'S':
            return 1
        if linha['Embarked'] == 'C':
            return 2
        if linha['Embarked'] == 'Q':
            return 3
        else:
            return 0

    df['Embarked'] = df.apply(embarqu, axis=1)

    def family(linha):
        if linha['SibSp'] > 0 or linha['Parch'] > 0:
            return 1
        else:
            return 0

    df['Family'] = df.apply(family, axis=1)

    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Title', 'Fare'], axis=1)

    return df

def prever_novos_dados(df_novos):
    import pandas as pd

    passenger_id = df_novos['PassengerId'].values

    model, scaler, imputer, columns, title_map = carregar_modelo()

    # Pré-processamento
    X = preprocessar_novos_dados(df_novos, title_map, columns)

    num_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Cabin_filled', 'Title_encoded', 'Family']

    X[num_cols] = imputer.transform(X[num_cols])

    # Escalonamento
    X_scaled = scaler.transform(X)

    # Predição
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    return pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': preds
        # ,'prob_survived': probs
    })

import pandas as pd
import openpyxl

novos_dados = pd.read_csv('../Bases/test.csv')

resultado = prever_novos_dados(novos_dados)

resultado.to_csv('Predicoes.csv', index=False)
