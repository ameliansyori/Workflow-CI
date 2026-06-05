import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_base_model():
    print("=== STARTING BASE MODELLING WITH MLFLOW AUTOLOG ===")
    
    mlflow.set_experiment("Heart_Disease_Experiment")
    mlflow.autolog()
    
    # 1. 🔥 FIX DARURAT: Membaca file utama hasil preprocessing
    main_data = pd.read_csv('dataset_preprocessing.csv')
    
    # Bersihkan baris kosong (NaN) jika ada
    main_data = main_data.dropna()
    
    # Split data otomatis secara instan (80% Train, 20% Test)
    train_data = main_data.sample(frac=0.8, random_state=42)
    test_data = main_data.drop(train_data.index)
    
    # Memisahkan fitur dan target
    X_train = train_data.drop(columns=['HeartDisease'])
    y_train = train_data['HeartDisease']
    X_test = test_data.drop(columns=['HeartDisease'])
    y_test = test_data['HeartDisease']
    
    # 2. Mulai Run MLflow Tracking
    with mlflow.start_run(run_name="Baseline_Random_Forest"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"-> [SUKSES] Model Dasar Selesai Dilatih. Akurasi Uji: {acc:.4f}")
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_base_model()
