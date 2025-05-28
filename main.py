from torchvision import transforms
from itertools import product

#Moje importy
from Classification.DumpCNN import DumpCNN
from Classification.DumpDataset import DumpDataset
from Classification.Training import ModelTrainer
from DataLoadAndProcessing import DataLoadClass


if __name__ == '__main__':
    '''
    pathDataFile30 = "2_refined_30.xlsx"
    pathDataFile35 = "2_refined_35.xlsx"
    pathAnotatedDataFile30 = "2_AnotatedData_refined_30.xlsx"
    pathAnotatedDataFile35 = "2_AnotatedData_refined_35.xlsx"
    pathImageOutputPath = "NewImageStorage30_15_6"
    dataPointsWidth = 30
    stride = 15
    minPointsForEvent = 6

    data_load_class = DataLoadClass(pathDataFile30, pathAnotatedDataFile30, pathImageOutputPath)
    data_load_class.loadAllData()
    data_load_class.exportDataAsImages(dataPointsWidth, stride, minPointsForEvent)

    data_load_class = DataLoadClass(pathDataFile35, pathAnotatedDataFile35, pathImageOutputPath)
    data_load_class.loadAllData()
    data_load_class.exportDataAsImages(dataPointsWidth, stride, minPointsForEvent)
    '''
    

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

    #data_path = "2_refined.xlsx"
    data_storage_path = "NewImageStorage"
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

    dump_dataset = DumpDataset(image_dir="NewImageStorage30_15_6", transform=transform)

    dropout_rates = [0.4]
    conv_drop_rates = [0.4]
    num_conv_layers = [4]
    hidden_dims = [256]
    learning_rates = [0.0005]
    use_pools = [True]

    for dr, ncl, hd, lr, up, cdr in product(dropout_rates, num_conv_layers, hidden_dims, learning_rates, use_pools, conv_drop_rates):
        model_params = {
            "input_shape": (3, image_pix, image_pix),
            "num_classes": 2,
            "dropout_rate": dr,
            "num_conv_layers": ncl,
            "hidden_dim": hd,
            "use_dropout": True,
            "use_pool": up,
            "conv_drop_rate": cdr
        }

        training_params = {
            "epochs": 100,
            "train_split": 0.8,
            "batch_size": 64,
            "learning_rate": lr,
            "patience": 10,
            "weight_decay": 1e-4
        }

        print(f"Training with: dr={dr}, conv={ncl}, hd={hd}, lr={lr}, pool={up}, cdr={cdr}")
        trainer = ModelTrainer (
            dataset=dump_dataset,
            model_class=DumpCNN,
            model_params=model_params,
            training_params=training_params,
            output_dir="NewTrainingResults"
        )
        trainer.train()