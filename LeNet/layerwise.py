import model
import wandb

wandb.init(project="quantization", entity="saiyam-jain")
batch_size = 1024
epochs = 30

wandb.config = {
    "epochs": epochs,
    "batch_size": batch_size,
}

activations = 1

if activations == 1:

    first_ab = "quantized_tanh(8,0)"
    second_ab = "quantized_tanh(8,0)"
    third_ab = "quantized_tanh(8,0)"
    fourth_ab = "quantized_tanh(8,0)"
    fifth_ab = "quantized_tanh(8,0)"

    print(first_ab, second_ab, third_ab, fourth_ab, fifth_ab)
    train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_ab=first_ab,
                                                                       second_ab=second_ab,
                                                                       third_ab=third_ab,
                                                                       fourth_ab=fourth_ab,
                                                                       fifth_ab=fifth_ab,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs)
    i = 10
    for first_ab in ["quantized_tanh(8,0)", "quantized_tanh(6,0)", "quantized_tanh(4,0)", "quantized_tanh(2,0)"]:
        i = i-2
        second_ab = "quantized_tanh(8,0)"
        third_ab = "quantized_tanh(8,0)"
        fourth_ab = "quantized_tanh(8,0)"
        fifth_ab = "quantized_tanh(8,0)"
        print(first_ab, second_ab, third_ab, fourth_ab, fifth_ab)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_ab=first_ab,
                                                                           second_ab=second_ab,
                                                                           third_ab=third_ab,
                                                                           fourth_ab=fourth_ab,
                                                                           fifth_ab=fifth_ab,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer activation bits": i,
            "second layer activation bits": 8,
            "third layer activation bits": 8,
            "fourth layer activation bits": 8,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })


    i = 10
    for second_ab in ["quantized_tanh(8,0)", "quantized_tanh(6,0)", "quantized_tanh(4,0)", "quantized_tanh(2,0)"]:
        i = i-2
        first_ab = "quantized_tanh(8,0)"
        third_ab = "quantized_tanh(8,0)"
        fourth_ab = "quantized_tanh(8,0)"
        fifth_ab = "quantized_tanh(8,0)"
        print(first_ab, second_ab, third_ab, fourth_ab, fifth_ab)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_ab=first_ab,
                                                                           second_ab=second_ab,
                                                                           third_ab=third_ab,
                                                                           fourth_ab=fourth_ab,
                                                                           fifth_ab=fifth_ab,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer activation bits": 8,
            "second layer activation bits": i,
            "third layer activation bits": 8,
            "fourth layer activation bits": 8,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

    i = 10
    for third_ab in ["quantized_tanh(8,0)", "quantized_tanh(6,0)", "quantized_tanh(4,0)", "quantized_tanh(2,0)"]:
        i = i-2
        first_ab = "quantized_tanh(8,0)"
        second_ab = "quantized_tanh(8,0)"
        fourth_ab = "quantized_tanh(8,0)"
        fifth_ab = "quantized_tanh(8,0)"
        print(first_ab, second_ab, third_ab, fourth_ab, fifth_ab)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_ab=first_ab,
                                                                           second_ab=second_ab,
                                                                           third_ab=third_ab,
                                                                           fourth_ab=fourth_ab,
                                                                           fifth_ab=fifth_ab,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer activation bits": 8,
            "second layer activation bits": 8,
            "third layer activation bits": i,
            "fourth layer activation bits": 8,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

    i = 10
    for fourth_ab in ["quantized_tanh(8,0)", "quantized_tanh(6,0)", "quantized_tanh(4,0)", "quantized_tanh(2,0)"]:
        i = i-2
        first_ab = "quantized_tanh(8,0)"
        second_ab = "quantized_tanh(8,0)"
        third_ab = "quantized_tanh(8,0)"
        fifth_ab = "quantized_tanh(8,0)"
        print(first_ab, second_ab, third_ab, fourth_ab, fifth_ab)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_ab=first_ab,
                                                                           second_ab=second_ab,
                                                                           third_ab=third_ab,
                                                                           fourth_ab=fourth_ab,
                                                                           fifth_ab=fifth_ab,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer activation bits": 8,
            "second layer activation bits": 8,
            "third layer activation bits": 8,
            "fourth layer activation bits": i,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

else:

    first_wb = 8
    second_wb = 8
    third_wb = 8
    fourth_wb = 8
    fifth_wb = 8
    for first_wb in [8, 6, 4, 2]:
        print(first_wb, second_wb, third_wb, fourth_wb, fifth_wb)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_wb=first_wb,
                                                                           second_wb=second_wb,
                                                                           third_wb=third_wb,
                                                                           fourth_wb=fourth_wb,
                                                                           fifth_wb=fifth_wb,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer quantization bits": first_wb,
            "second layer quantization bits": second_wb,
            "third layer quantization bits": third_wb,
            "fourth layer quantization bits": fourth_wb,
            "fifth layer quantization bits": fifth_wb,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

    first_wb = 8
    second_wb = 8
    third_wb = 8
    fourth_wb = 8
    fifth_wb = 8
    for second_wb in [8, 6, 4, 2]:
        print(first_wb, second_wb, third_wb, fourth_wb, fifth_wb)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_wb=first_wb,
                                                                           second_wb=second_wb,
                                                                           third_wb=third_wb,
                                                                           fourth_wb=fourth_wb,
                                                                           fifth_wb=fifth_wb,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer quantization bits": first_wb,
            "second layer quantization bits": second_wb,
            "third layer quantization bits": third_wb,
            "fourth layer quantization bits": fourth_wb,
            "fifth layer quantization bits": fifth_wb,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

    first_wb = 8
    second_wb = 8
    third_wb = 8
    fourth_wb = 8
    fifth_wb = 8
    for third_wb in [8, 6, 4, 2]:
        print(first_wb, second_wb, third_wb, fourth_wb, fifth_wb)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_wb=first_wb,
                                                                           second_wb=second_wb,
                                                                           third_wb=third_wb,
                                                                           fourth_wb=fourth_wb,
                                                                           fifth_wb=fifth_wb,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer quantization bits": first_wb,
            "second layer quantization bits": second_wb,
            "third layer quantization bits": third_wb,
            "fourth layer quantization bits": fourth_wb,
            "fifth layer quantization bits": fifth_wb,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

    first_wb = 8
    second_wb = 8
    third_wb = 8
    fourth_wb = 8
    fifth_wb = 8
    for fourth_wb in [8, 6, 4, 2]:
        print(first_wb, second_wb, third_wb, fourth_wb, fifth_wb)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_wb=first_wb,
                                                                           second_wb=second_wb,
                                                                           third_wb=third_wb,
                                                                           fourth_wb=fourth_wb,
                                                                           fifth_wb=fifth_wb,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer quantization bits": first_wb,
            "second layer quantization bits": second_wb,
            "third layer quantization bits": third_wb,
            "fourth layer quantization bits": fourth_wb,
            "fifth layer quantization bits": fifth_wb,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })

    first_wb = 8
    second_wb = 8
    third_wb = 8
    fourth_wb = 8
    fifth_wb = 8
    for fifth_wb in [8, 6, 4, 2]:
        print(first_wb, second_wb, third_wb, fourth_wb, fifth_wb)
        train_loss, train_accuracy, test_loss, test_accuracy = model.train(first_wb=first_wb,
                                                                           second_wb=second_wb,
                                                                           third_wb=third_wb,
                                                                           fourth_wb=fourth_wb,
                                                                           fifth_wb=fifth_wb,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs)

        wandb.log({
            "first layer quantization bits": first_wb,
            "second layer quantization bits": second_wb,
            "third layer quantization bits": third_wb,
            "fourth layer quantization bits": fourth_wb,
            "fifth layer quantization bits": fifth_wb,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy
        })
