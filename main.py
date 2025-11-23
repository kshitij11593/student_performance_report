import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model

def main():
    print("Loading data...")
    df = pd.read_csv("student_performance.csv")

    # ================================
    # 2. Correlation Heatmap
    # ================================
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


    print("Training model...")
    model, X_test, y_test = train_model(df)

    print("\nPredictions:")
    predictions = model.predict(X_test)

    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i]:.2f} | Actual: {y_test.iloc[i]}")
    
    
    # 1. Actual vs Predicted Graph
  
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, color='blue')
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title("Actual vs Predicted Student Performance")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
