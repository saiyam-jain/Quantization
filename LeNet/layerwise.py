import model
import wandb

wandb.init(project="quantization", entity="saiyam-jain")
batch_size = 256
epochs = 30

wandb.config = {
    "epochs": epochs,
    "batch_size": batch_size,
}

for fifth in [8, 6, 4, 2]:
    for fourth in [8, 6, 4, 2]:
        for third in [8, 6, 4, 2]:
            for second in [8, 6, 4, 2]:
                for first in [8, 6, 4, 2]:
                    wandb.log({
                        "first layer": first,
                        "second layer": second,
                        "third layer": third,
                        "fourth layer": fourth,
                        "fifth layer": fifth
                    })
                    print(first, second, third, fourth, fifth)
                    train_loss, train_accuracy, test_loss, test_accuracy = model.train(first,
                                                                                       second,
                                                                                       third,
                                                                                       fourth,
                                                                                       fifth,
                                                                                       batch_size=batch_size,
                                                                                       epochs=epochs)

                    wandb.log({
                        "Train Loss": train_loss,
                        "Train Accuracy": train_accuracy,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_accuracy
                    })
