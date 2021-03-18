import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class FirstModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 kernel_size = 3
                 ):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 64  # Set number of filters in first conv layer
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        # Define the convolutional layers
        print(f"Image chanels {image_channels}")
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels= 64,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels= 128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_output_features),
            nn.Linear(self.num_output_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(batch_size,-1)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class SecondModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 kernel_size = 3
                 ):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 128  # Set number of filters in first conv layer
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        # Define the convolutional layers
        print(f"Image chanels {image_channels}")
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels= 128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels= 128,
                out_channels= 256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_output_features),
            nn.Linear(self.num_output_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(batch_size,-1)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

class ThirdModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 kernel_size = 3
                 ):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 64  # Set number of filters in first conv layer
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        # Define the convolutional layers
        print(f"Image chanels {image_channels}")
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels= 64,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels= 64,
                out_channels= 128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4*4*64
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_output_features),
            nn.Linear(self.num_output_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(batch_size,-1)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def create_plots_acc(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(10, 8))
    plt.legend()
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def create_comp_plots(trainer: Trainer, trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(10, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss m1", npoints_to_average=10)
    utils.plot_loss(trainer2.train_history["loss"], label="Training loss m2", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss m1")
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss m2")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    first_model = SecondModel(image_channels=3, num_classes=10, kernel_size=3)
    second_model = ThirdModel(image_channels=3, num_classes=10, kernel_size=5)
    first_trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        first_model,
        dataloaders
    )
    second_trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        second_model,
        dataloaders
    )
    first_trainer.train()
    first_trainer.load_best_model()
    second_trainer.train()
    second_trainer.load_best_model()

    print(f"Final training loss and  accuracy {compute_loss_and_accuracy(first_trainer.dataloader_train, first_trainer.model, first_trainer.loss_criterion)}")
    print(f"Final validation loss and accuracy {compute_loss_and_accuracy(first_trainer.dataloader_val, first_trainer.model, first_trainer.loss_criterion)}")
    print(f"Final test loss and accuracy {compute_loss_and_accuracy(first_trainer.dataloader_test, first_trainer.model, first_trainer.loss_criterion)}")

    create_plots_acc(first_trainer, "task3e - with transforms - test")
    create_comp_plots(first_trainer, second_trainer, "task3d")
