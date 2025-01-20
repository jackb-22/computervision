import pandas as pd
import pickle

def load_model():
    with open('best_polynomial_svm.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    input_data = df.values
    predictions = model.predict(input_data)
    prediction_labels = ["Car" if pred == 0 else "Truck" for pred in predictions]
    return prediction_labels

if __name__ == "__main__":
    model = load_model()
    csv_file = input("Enter your CSV file path here:")
    predictions = predict_from_csv(csv_file)
    
    for i, pred in enumerate(predictions, start=1):
        print(f"Row {i}: {pred}")
