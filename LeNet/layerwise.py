import model
import wandb

wandb.init(project="quantization", entity="saiyam-jain")
batch_size = 1024
epochs = 30

wandb.config = {
    "epochs": epochs,
    "batch_size": batch_size,
}

first = "quantized_relu(8,0)"
second = "quantized_relu(8,0)"
third = "quantized_relu(8,0)"
fourth = "quantized_relu(8,0)"
fifth = "quantized_relu(8,0)"

print(first, second, third, fourth, fifth)
train_loss, train_accuracy, test_loss, test_accuracy = model.train(first,
                                                                   second,
                                                                   third,
                                                                   fourth,
                                                                   fifth,
                                                                   batch_size=batch_size,
                                                                   epochs=epochs)
i = 8
for first in ["quantized_relu(6,0)", "quantized_relu(4,0)", "quantized_relu(2,0)"]:
    i = i-2
    second = "quantized_relu(8,0)"
    third = "quantized_relu(8,0)"
    fourth = "quantized_relu(8,0)"
    fifth = "quantized_relu(8,0)"
    print(first, second, third, fourth, fifth)
    train_loss, train_accuracy, test_loss, test_accuracy = model.train(first,
                                                                       second,
                                                                       third,
                                                                       fourth,
                                                                       fifth,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs)

    wandb.log({
        "first layer quantization bits": i,
        "second layer quantization bits": 8,
        "third layer quantization bits": 8,
        "fourth layer quantization bits": 8,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })

i = 8
for second in ["quantized_relu(6,0)", "quantized_relu(4,0)", "quantized_relu(2,0)"]:
    i = i - 2
    first = "quantized_relu(8,0)"
    third = "quantized_relu(8,0)"
    fourth = "quantized_relu(8,0)"
    fifth = "quantized_relu(8,0)"
    print(first, second, third, fourth, fifth)
    train_loss, train_accuracy, test_loss, test_accuracy = model.train(first,
                                                                       second,
                                                                       third,
                                                                       fourth,
                                                                       fifth,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs)

    wandb.log({
        "first layer quantization bits": 8,
        "second layer quantization bits": i,
        "third layer quantization bits": 8,
        "fourth layer quantization bits": 8,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })

i = 8
for third in ["quantized_relu(6,0)", "quantized_relu(4,0)", "quantized_relu(2,0)"]:
    i = i - 2
    first = "quantized_relu(8,0)"
    second = "quantized_relu(8,0)"
    fourth = "quantized_relu(8,0)"
    fifth = "quantized_relu(8,0)"
    print(first, second, third, fourth, fifth)
    train_loss, train_accuracy, test_loss, test_accuracy = model.train(first,
                                                                       second,
                                                                       third,
                                                                       fourth,
                                                                       fifth,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs)

    wandb.log({
        "first layer quantization bits": 8,
        "second layer quantization bits": 8,
        "third layer quantization bits": i,
        "fourth layer quantization bits": 8,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })

i = 8
for fourth in ["quantized_relu(6,0)", "quantized_relu(4,0)", "quantized_relu(2,0)"]:
    i = i - 2
    first = "quantized_relu(8,0)"
    second = "quantized_relu(8,0)"
    third = "quantized_relu(8,0)"
    fifth = "quantized_relu(8,0)"
    print(first, second, third, fourth, fifth)
    train_loss, train_accuracy, test_loss, test_accuracy = model.train(first,
                                                                       second,
                                                                       third,
                                                                       fourth,
                                                                       fifth,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs)

    wandb.log({
        "first layer quantization bits": 8,
        "second layer quantization bits": 8,
        "third layer quantization bits": 8,
        "fourth layer quantization bits": i,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })
