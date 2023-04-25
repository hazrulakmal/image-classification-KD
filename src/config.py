class configurations:
    num_workers = 4
    seed = 42
    in_features = 512
    name_run = "resnet18-vanilla-lightning"
    model = "resnet18"
    learning_rate = 0.01
    betas = (0.9, 0.999)
    eps = 1e-6
    epochs = 10
    accelerator = "gpu"
    cloud_compute = "gpu"


