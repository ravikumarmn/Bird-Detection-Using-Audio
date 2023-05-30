import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from data_loader import train_loader, val_loader
from model import BirdClassifier
from utils import eval_fn, plot_confusion_matrix, train_fn

# import wandb


result_dir = "webapp/static/result"
os.makedirs(result_dir, exist_ok=True)


# def start_wandb():
#     wandb.init(
#     # set the wandb project where this run will be logged
#     project="Bird-classification",
#     name = config.SAVE_DIR_PATH,
#     # track hyperparameters and run metadata
#     config={
#         k:v for k,v in config.__dict__.items() if "__" not in k
#     }
# )

# if config.WANDB:
#     start_wandb()


def train(params, model, train_loader, val_loader, optimizer, loss_fn):
    train_losses = []
    train_accuracy = []
    eval_losses = []
    eval_accuracy = []

    best_eval_loss = float('inf')
    patience = params["PATIENCE"]
    early_stop_counter = 0

    if params['DEBUG']:
        print("Running in Debug Mode")

    print("Training data size:", len(train_loader) * params['BATCH_SIZE'])
    print("Validation data size:", len(val_loader) * params['BATCH_SIZE'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)

    # Cross-validation
    kf = KFold(n_splits=params['NUM_FOLDS'], shuffle=True, random_state=params['RANDOM_STATE'])
    fold = 1

    for train_index, val_index in kf.split(train_loader.dataset):
        print(f"Fold: {fold}/{params['NUM_FOLDS']}")
        fold += 1

        # Split train and validation dataset
        train_dataset = torch.utils.data.Subset(train_loader.dataset, train_index)
        val_dataset = torch.utils.data.Subset(train_loader.dataset, val_index)

        # Create new data loaders for the current fold
        train_loader_fold = torch.utils.data.DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)

        epochs = list(range(1, params['EPOCHS'] + 1))

        # Create a list to store learning rates at each epoch
        lr_rates = []

        for epoch in epochs:
            # Training
            train_loss, train_acc = train_fn(model, train_loader_fold, optimizer, loss_fn, params['DEVICE'], params['L2_LAMBDA'])
            train_losses.append(train_loss)
            train_accuracy.append(train_acc)

            eval_loss, eval_acc = eval_fn(model, val_loader_fold, loss_fn, params['DEVICE'])
            eval_losses.append(eval_loss)
            eval_accuracy.append(eval_acc)
            # wandb.log(
            #     {
            #         "train_accuracy":train_acc,
            #         "validation_accuracy":eval_acc,
            #         "epoch" : epoch
            #     }
            # )

            # wandb.log(
            #     {
            #         "train_loss":train_loss,
            #         "validation_loss":eval_loss,
            #         "epoch" : epoch
            #     }
            # )

            # Print train and validation loss and accuracy
            print(f"Epoch {epoch}/{params['EPOCHS']}")
            print(f"Train Loss: {train_loss:.4f}\tTrain Accuracy: {train_acc:.4f}")
            print(f"Val Loss: {eval_loss:.4f}\tVal Accuracy: {eval_acc:.4f}")
            print()

            if not params["DEBUG"]:
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    # Save checkpoint
                    model_state_dict = model.state_dict()
                    torch.save(model_state_dict, params["SAVE_DIR_PATH"])
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Validation loss has not improved for {patience} epochs. Training stopped.")
                        break
            scheduler.step(eval_loss)

            # Store the learning rate at each epoch
            lr_rates.append(optimizer.param_groups[0]['lr'])

#             wandb.log(
# {
#                 "train_loss_fold":torch.tensor(train_losses).mean(),
#                 "validation_loss_fold":torch.tensor(eval_losses).mean(),
#                 "train_acc_fold":torch.tensor(train_accuracy).mean(),
#                 "validation_acc_fold":torch.tensor(eval_accuracy).mean(),
#           }  )
        
        # Plotting the learning rate scheduler
        plt.figure(figsize=(10, 5))
        plt.plot(epochs[:len(lr_rates)], lr_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Scheduler')
        plt.savefig(os.path.join(result_dir, f"learning_rate_plot_fold{fold}.png"))
        plt.close()

    # Plotting the train and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs[:len(train_losses)], train_losses, label='Training Loss')
    plt.plot(epochs[:len(eval_losses)], eval_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss_plot.png"))
    plt.close()

    # Plotting the train and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs[:len(train_accuracy)], train_accuracy, label='Training Accuracy')
    plt.plot(epochs[:len(eval_accuracy)], eval_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_dir, "accuracy_plot.png"))
    plt.close()

    # Generate and save the confusion matrix
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch_waveform = batch["spectrogram"].to(params['DEVICE'])
            batch_labels = batch["label"].to(params['DEVICE'])

            outputs = model(batch_waveform)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(batch_labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    plot_confusion_matrix(true_labels, pred_labels, classes=["American Robin", "Bewick's Wren", "Northern Cardinal", "Northern Mockingbird", "Song Sparrow"])



def main():
    params = {k:v for k,v in config.__dict__.items() if "__" not in k}
    print(f"Using {params['DEVICE']} Device.")
    model = BirdClassifier(num_classes=params["NUM_CLASSES"]).to(params['DEVICE'])
    optimizer = Adam(model.parameters(), lr= params["LEARNING_RATE"])
    loss_fn = nn.CrossEntropyLoss()

    train(params, model, train_loader, val_loader, optimizer, loss_fn)
    # wandb.finish()
if __name__=="__main__":
    main()
    