import utils
import matplotlib.pyplot as plt
from task4mod import pre_process_images, one_hot_encode, SoftmaxModel, calculate_mean_and_std
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean, std = calculate_mean_and_std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    #num_epochs = 5

    model_two = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_two = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_two, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )


    train_history, val_history = trainer.train(num_epochs)
    train_history_two, val_history_two = trainer_two.train(num_epochs)

    plt.figure(figsize=(12,9))
    #plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"], "Base model task 3 ", npoints_to_average =10)
    utils.plot_loss(train_history_two["loss"], "10 hidden layer", npoints_to_average = 10)
    plt.ylim([-0.05, 0.4])
    plt.ylabel("Training loss")
    plt.legend()
    #plt.subplot(1, 2, 2)
    #plt.ylim([0.85, 1.0])
    #utils.plot_loss(train_history["accuracy"], "Training set - 128")
    #utils.plot_loss(val_history["accuracy"], "Validation set - 128")
    #plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4e_train_loss.png", dpi = 600)
