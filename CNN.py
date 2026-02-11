import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np

## cpu yada gpu seçmek için 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

## data base
BASE_DIR = r"C:\Users\Motasem\Desktop\ML-MOTASEM YILDIZ\archive\Final Dataset"

## data basin içindeki dosylar traning , validation , testing 

## traning modeli eğitmek için kullandığımız dosya
TRAIN_DIR = os.path.join(BASE_DIR, "training")
## validation dosyası modelin eğitimi sırasında hiç görmediği modelin performansını ölçmek için kullanıyoruz  
## birde overfitting engellemek için kullaniyoruz
## yani model anlıyormu yoksa sadece data basi mi ezberliyor ölçmek için
VAL_DIR   = os.path.join(BASE_DIR, "validation")

## testing dosyasi tabi herhangi bir resim gertirebiliriz sadece bu dosyadaki resimler degil
TEST_DIR  = os.path.join(BASE_DIR, "testing")


## Hyperparameters modelin nasıl eğiteceğmizin ayarları 

## bütün resimler 224*224 pixele çevirecağız çünkü cnn farklı farklı hacımlarla eğitmek 
# bazi sorunlar çıkartabilir
IMG_SIZE = 224

## model resimler tek tek almiyor batch grup olarak aliyor
## her batchta kaç resim alıyor
BATCH_SIZE = 16

## model kaç defa resimleri görüyör
NUM_EPOCHS = 10

## learning rate 0.001 daha büyük olursa eğitimde büyük atlamalar yapacak küçük olursa
## küçük olursa çok yavaş bi şekilde eğitelecek
LEARNING_RATE = 1e-3

## normalization işlemi 
## burda normalization resimlerdeki pixel rakamlari değişteriyor
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]


## training başlamadan önce resimleri yaptığımız diğerlerle hazırlamak
## transform
## Augmentation var yani resmin birkaç kopyası oluştutyor ama her biri eskisinder biraz daha gelişmiş halde
## resimleri ezberlememek için
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), ## sağ sol
    transforms.RandomRotation(10),## rotation
    transforms.ToTensor(), ## pytoch direk resimlerde çalışmıyor tensorlerle çalışyor
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

## transform validation için
## Augmentation yok
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

## datayı yükelemek 
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=VAL_DIR,   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=val_test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


def create_model(num_classes: int):
    # modeli cnn olarak burda oluşturuyoruz
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

## modele cpuyua göndermek 
model = create_model(num_classes).to(device)

## lose function 
criterion = nn.CrossEntropyLoss()

## sadece son layeri eğitiyoruz diğer layerleri döndürdük
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

## eğitme işlemi bu fonksiyonda yapıyoruz
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0.0 ## en iyi accuary sakliyoruz

    for epoch in range(num_epochs): 
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device) # batch batch alip modele ve cpuya gönderiyoruz
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()


            ## her epochun acuuary ve loss hesapliyoruz
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects / total_train

        ## eğitildikten sonra kontrol etmek için validaition moduna geçiyoruz
        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_acc = val_corrects / val_total ## validationin en iyi accuary hesapliyoruz

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"- Train Loss: {epoch_loss:.4f} "
            f"- Train Acc: {epoch_acc:.4f} "
            f"- Val Acc: {val_acc:.4f}")
        
        ## eğitildikten sonra modelin en iyi ayrı bir dosyada saklayacağız tabi bunu validation kullanarak 
        # yapıyoruz 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_model.pth")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

## eğitimi değernlerdimek için
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    ## değerlendirmek içim confusıon matrix 
    ## presicion recall f1 score 
    ## weghited f1score hesapliyoruz

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (CNN):")
    print(cm)

    print("\nClassification Report (CNN):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"\nWeighted F1-score (CNN): {f1:.4f}")

if __name__ == "__main__":
    print("Starting CNN training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

    print("\nLoading best model and evaluating on test set...")
    best_model = create_model(num_classes).to(device)
    best_model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))

    evaluate_model(best_model, test_loader, class_names)


