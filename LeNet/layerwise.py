import model
# import wandb
#
# wandb.init(project="quantization", entity="saiyam-jain")
batch_size = 256
epochs = 30

# wandb.config = {
#     "epochs": epochs,
#     "batch_size": batch_size,
#     "lr": lr
# }

# for fifth in [8, 6, 4, 2]:
#     for fourth in [8, 6, 4, 2]:
#         for third in [8, 6, 4, 2]:
#             for second in [8, 6, 4, 2]:
#                 for first in [8, 6, 4, 2]:
#                     wandb.log({
#                         "first layer": first,
#                         "second layer": second,
#                         "third layer": third,
#                         "fourth layer": fourth,
#                         "fifth layer": fifth
#                     })

# wandb.log({
#     "Epoch": epoch + 1,
#     "Train Loss": train_loss.result().numpy(),
#     "Train Accuracy": train_accuracy.result().numpy(),
#     "Test Loss": test_loss.result().numpy(),
#     "Test Accuracy": test_accuracy.result().numpy()
# })

train_loss, train_accuracy, test_loss, test_accuracy = model.train(8, 8, 8, 8, 8, epochs=10)
print(train_loss, train_accuracy, test_loss, test_accuracy)
train_loss, train_accuracy, test_loss, test_accuracy = model.train(2, 2, 2, 2, 2, epochs=10)
print(train_loss, train_accuracy, test_loss, test_accuracy)
