import data_ingestion
import scoring
import training


def main():
    data_ingestion.fetch_housing_data()
    housing = data_ingestion.load_housing_data()

    X, y = training.preprocess_data(housing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = training.train_model(X_train, y_train)

    rmse = scoring.evaluate_model(model, X_test, y_test)
    print("Root Mean Squared Error:", rmse)


if __name__ == "__main__":
    main()
