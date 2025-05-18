from torchvision import transforms
from itertools import product

#Moje importy
from Classification.DumpCNN import DumpCNN
from Classification.DumpDataset import DumpDataset
from Classification.Training import ModelTrainer


if __name__ == '__main__':
    pathDataFile30 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_refined_30.xlsx"
    pathDataFile35 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_refined_35.xlsx"
    pathAnotatedDataFile30 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_AnotatedData_refined_30.xlsx"
    pathAnotatedDataFile35 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_AnotatedData_refined_35.xlsx"
    pathImageOutputPath = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\3VygenerovaneSubory"
    dataPointsWidth = 60
    stride = 20
    minPointsForEvent = 10

    data_load_class = DataLoadClass(pathDataFile30, pathAnotatedDataFile30, pathImageOutputPath)
    data_load_class.loadAllData()
    data_load_class.exportDataAsImages(dataPointsWidth, stride, minPointsForEvent)

    # data_load_class = DataLoadClass(pathDataFile35, pathAnotatedDataFile35, pathImageOutputPath)
    # data_load_class.loadAllData()
    # data_load_class.exportDataAsImages(dataPointsWidth, stride, minPointsForEvent)


# -------------------------------------
# 50
# 25
# 10
# 1100 negativnych 100 pozitivnych

#---------------------------------------
# 20
# 5
# 10
# 5788 negativnych 203 pozitivnych

#---------------------------------------
# 100
# 20
# 20
# 1447 negativnych 47 pozitivnych

#---------------------------------------
# 250
# 75
# 20
# 369 negativnych 24 pozitivnych 4 nevygenerovane

#---------------------------------------
# 200
# 100
# 10
# 236 negativnych 35 pozitivnych 27 nevygenerovane

#---------------------------------------
# 60
# 20
# 10
# 1349 negativnych 145 pozitivnych 2 nevygenerovane
    data_path = "2_refined.xlsx"
    data_storage_path = "ImageStorage"
    #data_load_class = DataLoadClass(data_path, data_storage_path, 2.0)

    #data_load_class.loadData()
    #data_load_class.normalizeData()
    #data_load_class.exportDataAsImages('Red', 'Green', 'Blue')
    

    image_pix = 256
    transform = transforms.Compose([
        transforms.Resize((image_pix, image_pix)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    
    dump_dataset = DumpDataset(image_dir=data_storage_path, transform=transform)
    
    model_params = {
        "input_shape": (3, image_pix, image_pix),
        "num_classes": 2,
        "dropout_rate": 0.45,
        "num_conv_layers": 3,
        "hidden_dim": 256,
        "use_dropout": True,
        "use_pool": True
    }
    
    training_params = {
        "epochs": 100,
        "train_split": 0.8,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "patience": 10
    }
    
    print("vytvaram trainer")
    trainer = ModelTrainer(
        dataset=dump_dataset, 
        model_class=DumpCNN, 
        model_params=model_params, 
        training_params=training_params
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0
    
    plt.ion()
    fig, ax = plt.subplots()
    train_losses = []
    validation_losses = []
    line1, = ax.plot([], [], 'orange', label='Training loss')
    line2, = ax.plot([], [], 'blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/100], Train Loss: {epoch_loss:.4f}")


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        train_losses.append(epoch_loss)
        validation_losses.append(val_loss)

        line1.set_data(range(len(train_losses)), train_losses)
        line2.set_data(range(len(validation_losses)), validation_losses)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping!")
            break
        
    model.load_state_dict(torch.load("best_model.pth"))        
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    plt.ioff()
    plt.show()

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("zacinam trenovat")
    trainer.train()