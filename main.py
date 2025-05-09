from torchvision import transforms

#Moje importy
from Classification.DumpCNN import DumpCNN
from Classification.DumpDataset import DumpDataset
from Classification.Training import ModelTrainer


if __name__ == '__main__':
    data_path = "2_refined.xlsx"
    data_storage_path = "ImageStorage"
    #data_load_class = DataLoadClass(data_path, data_storage_path, 2.0)

    #data_load_class.loadData()
    #data_load_class.normalizeData()
    #data_load_class.exportDataAsImages('Red', 'Green', 'Blue')
    
    image_pix = 256
    
    transform = transforms.Compose([
        #cierno biele
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_pix, image_pix)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
        #cierno biele
        #transforms.Normalize(mean=[0.5], std=[0.5]),
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
    
    print("zacinam trenovat")
    trainer.train()
