import pandas as pd
from model import train_model

def main():
    print("Loading data...")
    df = pd.read_csv("data.csv")

    print("Training model...")
    model, X_test, y_test = train_model(df)

    print("\nPredictions:")
    predictions = model.predict(X_test)

    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i]:.2f} | Actual: {y_test.iloc[i]}")

if __name__ == "__main__":
    main()
