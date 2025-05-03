import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet


def load_data(path):
    return pd.read_csv(path)


def clip_outliers(df, threshold=5):
    """
    Clip numerical features to within mean ± threshold * std
    """
    df_clip = df.copy()
    for col in df_clip.select_dtypes(include=[np.number]).columns:
        lower = df_clip[col].mean() - threshold * df_clip[col].std()
        upper = df_clip[col].mean() + threshold * df_clip[col].std()
        df_clip[col] = df_clip[col].clip(lower, upper)
    return df_clip


def target_transform(y):
    return np.log1p(y)

def target_inverse(y_t):
    return np.expm1(y_t)


def build_preprocessor(numeric_cols, categorical_cols, scaler='standard'):
    """
    Construct ColumnTransformer for numerical and categorical preprocessing
    """
    if scaler == 'robust':
        num_scaler = RobustScaler()
    else:
        num_scaler = StandardScaler()

    preproc = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', num_scaler)
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])
    return preproc


def train_approach_C(train_path):
    # Load and split data
    df = load_data(train_path)
    X = df.drop('Scoville Heat Units (SHU)', axis=1)
    y = df['Scoville Heat Units (SHU)']
    y_bin = (y > 0).astype(int)

    # Identify columns
    num_cols = X.select_dtypes(include=[np.number]).columns.to_list()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.to_list()

    # Preprocessing
    base_prep = build_preprocessor(num_cols, cat_cols, scaler='robust')
    # Add polynomial features for regression
    poly_prep = Pipeline([
        ('prep', base_prep),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])

    # Classification pipeline (best params from notebook: n_estimators=150, max_depth=None)
    clf_pipe = Pipeline([
        ('prep', base_prep),
        ('clf', RandomForestClassifier(random_state=42, n_estimators=150, max_depth=None))
    ])
    clf_pipe.fit(X, y_bin)

    # Regression pipeline on positive targets (best params: alpha=1.0, l1_ratio=0)
    mask = y > 0
    reg_pipe = Pipeline([
        ('prep', poly_prep),
        ('reg', ElasticNet(max_iter=200000, alpha=1.0, l1_ratio=0))
    ])
    reg_pipe.fit(X[mask], target_transform(y[mask]))

    return clf_pipe, reg_pipe


def predict_approach_C(test_path, clf_pipe, reg_pipe):
    test_df = load_data(test_path)
    # Encode categories as integer codes
    for c in test_df.select_dtypes(exclude=[np.number]).columns:
        test_df[c] = test_df[c].astype('category').cat.codes
    test_df = clip_outliers(test_df, threshold=5)

    y_bin = clf_pipe.predict(test_df)
    y_reg_log = reg_pipe.predict(test_df)
    y_reg = target_inverse(y_reg_log)

    return np.where(y_bin == 0, 0, y_reg)


if __name__ == '__main__':
    clf_C, reg_C = train_approach_C('assets/train.csv')
    preds = predict_approach_C('assets/test.csv', clf_C, reg_C)
    output_df = pd.DataFrame({'Scoville Heat Units (SHU)': preds}, index=load_data('assets/test.csv').index)
    output_df.index.name = 'index'
    output_df.to_csv('assets/test_predictions_C.csv')
    print("Saved assets/test_predictions_C.csv")
