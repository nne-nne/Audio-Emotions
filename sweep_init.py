import wandb

if __name__ == "__main__":
    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_accuracy',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        "layer_size_1": {
            'values': [128]
        },
        "layer_size_2": {
            'values': [64]
        },
        'layer_size_3': {
            'values': [64, 128, 256]
        },
        "activation_3": {
            'values': ['relu']
        },
        'dropout': {
            'values': [0.2, 0.4]
        },
        "activation_4": {
            'values': ['softmax']
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        "loss": {
            'values': ['categorical_crossentropy']
        },
        "metric": {
            'values': ['accuracy']
        },
        'epoch': {
            'values': [10, 20, 30]
        },
        "batch_size": {
            'values': [32]
        }
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='audioemo')