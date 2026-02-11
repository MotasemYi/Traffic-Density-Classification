import os
import numpy as np
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib  

BASE_DIR = r"C:\Users\Motasem\Desktop\ML-MOTASEM YILDIZ\archive\Final Dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "training")
VAL_DIR   = os.path.join(BASE_DIR, "validation")
TEST_DIR  = os.path.join(BASE_DIR, "testing")

IMG_SIZE = 64  

simple_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

## bu fonksiyon dosyadan resimleri alır ve x ve y oluşturur
## x featurelar (resmin pixel değerleri)
## y label (class)
def build_xy(folder):
    dataset = datasets.ImageFolder(root=folder, transform=simple_transform)
    X_list, y_list = [], []

    for img, label in dataset:
        ## flttan yani resmi 3d array dan rakamlardan oluşan bir vectora çeviriyoruz önemli bi nokta bu
        ## yani resmi tek bir uzun vector yapıyoruz
        X_list.append(img.view(-1).numpy())  
        y_list.append(label)
    ## listeyi numpy array'e çeviriyoruz
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, dataset.classes

if __name__ == "__main__":
    
    X_train, y_train, class_names = build_xy(TRAIN_DIR)
    X_val,   y_val,   _           = build_xy(VAL_DIR)
    X_test,  y_test,  _           = build_xy(TEST_DIR)

    ## training + validation birleştiriyoruz
    ## çünkü final modeli daha fazla data ile eğitmek istiyoruz
    X_tr_full = np.concatenate([X_train, X_val], axis=0)
    y_tr_full = np.concatenate([y_train, y_val], axis=0)

    
    scaler = StandardScaler()
    X_tr_full_scaled = scaler.fit_transform(X_tr_full)
    X_test_scaled    = scaler.transform(X_test)

    ## Random Forest modeli oluşturuyoruz
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    print("Training Random Forest...")
    ## modeli eğitiyoruz
    rf.fit(X_tr_full_scaled, y_tr_full)

    ## test datada tahmin yapıyoruz
    y_pred = rf.predict(X_test_scaled)

    ## değerlendirmek içim confusıon matrix 
    ## presicion recall f1 score 
    ## weghited f1score hesapliyoruz

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (Random Forest):")
    print(cm)

    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, target_names=class_names))

    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nWeighted F1-score (Random Forest): {f1:.4f}")

    
    joblib.dump(rf, "rf_model.joblib")
    joblib.dump(scaler, "rf_scaler.joblib")
    print("\nSaved rf_model.joblib and rf_scaler.joblib")
