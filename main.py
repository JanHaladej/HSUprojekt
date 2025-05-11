from torchvision import transforms
from itertools import product

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
        transforms.Resize((image_pix, image_pix)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dump_dataset = DumpDataset(image_dir="ImageStorage", transform=transform)

    dropout_rates = [0.4]
    num_conv_layers = [4]
    hidden_dims = [256]
    learning_rates = [0.0005]
    use_pools = [True]

    for dr, ncl, hd, lr, up in product(dropout_rates, num_conv_layers, hidden_dims, learning_rates, use_pools):
        model_params = {
            "input_shape": (3, image_pix, image_pix),
            "num_classes": 2,
            "dropout_rate": dr,
            "num_conv_layers": ncl,
            "hidden_dim": hd,
            "use_dropout": True,
            "use_pool": up
        }

        training_params = {
            "epochs": 100,
            "train_split": 0.7,
            "batch_size": 32,
            "learning_rate": lr,
            "patience": 10
        }

        print(f"Training with: dr={dr}, conv={ncl}, hd={hd}, lr={lr}, pool={up}")
        trainer = ModelTrainer(
            dataset=dump_dataset,
            model_class=DumpCNN,
            model_params=model_params,
            training_params=training_params
        )
        trainer.train()
