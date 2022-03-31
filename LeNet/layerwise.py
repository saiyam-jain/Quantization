import model
import wandb

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

train_loss, train_accuracy, test_loss, test_accuracy = model.train(8, 8, 8, 8, 8, epochs=2)
print(train_loss, train_accuracy, test_loss, test_accuracy)
train_loss, train_accuracy, test_loss, test_accuracy = model.train(6, 6, 6, 6, 6, epochs=2)
print(train_loss, train_accuracy, test_loss, test_accuracy)