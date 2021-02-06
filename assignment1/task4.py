import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """

    # forward pass
    logits = model.forward(X)

    # finding the index of the max values for both arrays
    logits = logits.argmax(axis=1)
    targets = targets.argmax(axis=1)

    # counting the equal entries and averaging
    accuracy = np.count_nonzero(np.equal(targets, logits)) / X.shape[0]

    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """

        logits = self.model.forward(X_batch)
        self.model.backward(X_batch, logits, Y_batch)
        self.model.update_weights(self.learning_rate)
        loss = cross_entropy_loss(Y_batch, logits)

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True
    early_stopping = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.


    # 4b)

    # Intialize model
    model_one = SoftmaxModel(l2_reg_lambda)
    trainer_one = SoftmaxTrainer(
        model_one, learning_rate, batch_size, shuffle_dataset,
        early_stopping, X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer_one.train(num_epochs)


    # Train a model with L2 regularization (task 4b)

    model_two = SoftmaxModel(l2_reg_lambda=1.0)
    trainer_two = SoftmaxTrainer(
        model_two, learning_rate, batch_size, shuffle_dataset,
        early_stopping, X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer_two.train(num_epochs)

    im_h = 28
    im_w = 28

    weight_one = model_one.w.T[:, :-1].reshape((-1, im_h, im_w))
    weight_two = model_two.w.T[:, :-1].reshape((-1, im_h, im_w))

    weight_one = np.hstack(weight_one)
    weight_two = np.hstack(weight_two)

    weights = np.vstack((weight_one, weight_two))


    # Plotting of softmax weights (Task 4b)
    plt.imsave("task4b_softmax_weight.png", weights, cmap="gray")
    plt.close()

    # Array for yask 4e)
    weight_norm = []
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]

    # setting the graph to the an expected range
    plt.ylim([.70, .93])

    for l2 in l2_lambdas:
        model = SoftmaxModel(l2)
        trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        early_stopping, X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

        # recording norm for 4e)
        print(float(np.sum(model.w*model.w)))
        weight_norm.append(np.sum(model.w*model.w))

        # plotting for task 4c)
        utils.plot_loss(val_history["accuracy"].copy(), f"lambda: {l2}")
        print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:", cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))


    # Plotting of accuracy for difference values of lambdas (task 4c)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation accuracy - Average")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.close()


    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_lambdas, weight_norm)

    plt.xlabel("Lambda")
    plt.ylabel("L2 norm")

    plt.savefig("task4e_l2_reg_norms.png")
