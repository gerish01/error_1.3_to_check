import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(data):
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return data.drop("median_house_value", axis=1), data["median_house_value"]


def prepare_pipeline():
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("encoder", OneHotEncoder()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            (
                "num",
                num_pipeline,
                [
                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("cat", cat_pipeline, ["ocean_proximity"]),
        ]
    )

    return full_pipeline


def train_model(X_train, y_train):
    pipeline = prepare_pipeline()
    forest_reg = RandomForestRegressor(random_state=42)
    full_pipeline_with_predictor = Pipeline(
        [
            ("preparation", pipeline),
            ("forest_reg", forest_reg),
        ]
    )
    full_pipeline_with_predictor.fit(X_train, y_train)
    return full_pipeline_with_predictor
