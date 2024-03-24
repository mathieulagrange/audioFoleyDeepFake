from bs4 import StopParsing
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tqdm import trange
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import optim 



# Function to process embeddings
def process_embeddings(directory, embeddings_files, label, system_id = None, baseline =False, track_included=False, folder_paths=None):
    data = []
    if folder_paths is None:
        folder_paths = [directory]

    for embedding_name in embeddings_files:
        print(f"----------->Processing {embedding_name} embeddings<-------------")
        for base_path in folder_paths:
            
                if track_included:
                    for track in os.listdir(base_path) :
                        track_folder_path = os.path.join(base_path,track)
                        print(os.listdir(track_folder_path))
                        for system_id in os.listdir(track_folder_path):
                            print(f" Processing {system_id} embeddings")
                            system_folder_path = os.path.join(track_folder_path,system_id)

                            for class_sound in os.listdir(system_folder_path) :
                                class_folder_path = os.path.join(system_folder_path, class_sound)
                                embeddings_folder_path = os.path.join(class_folder_path, f"embeddings/{embedding_name}")
                                if os.path.exists(embeddings_folder_path):
                                    for embedding_file in tqdm(os.listdir(embeddings_folder_path)):
                                        embedding_file_path = os.path.join(embeddings_folder_path, embedding_file)
                                        embedding = np.load(embedding_file_path)
                                        
                                        data.append({
                                            'class': class_sound,
                                            'embedding': embedding,
                                            'embedding_type': embedding_name,
                                            'label': label,
                                            'system_id': system_id,
                                            'track': track,
                                            'path_file': embedding_file_path
                                            })
                else :
                    track = None

                    if baseline :
                        system_id = 'Baseline'
                        
                    for class_sound in os.listdir(base_path):
                        class_folder_path = os.path.join(base_path, class_sound)
                        embeddings_folder_path = os.path.join(class_folder_path, f"embeddings/{embedding_name}")
                        if os.path.exists(embeddings_folder_path):

                            for embedding_file in tqdm(os.listdir(embeddings_folder_path), desc=f"Processing {class_sound}"):
                                embedding_file_path = os.path.join(embeddings_folder_path, embedding_file)
                                embedding = np.load(embedding_file_path)
                                data.append({
                                    'class': class_sound,
                                    'embedding': embedding,
                                    'embedding_type': embedding_name,
                                    'label': label,
                                    'system_id': system_id,
                                    'track': track,
                                    'path_file': embedding_file_path
                                    })
    return pd.DataFrame(data)




# Class for model handlings
class ModelTrainer:
    """
     Class for training, validation, and prediction operations for a PyTorch model.

    Attributes:
        model (torch.nn.Module): The neural network model to train and evaluate.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_function: The loss function to use for evaluating model performance.
        hparams: Hyperparameters used for training and validation.
        scheduler (optional): Learning rate scheduler.
        early_stop (bool): Whether to stop training early if validation performance degrades.
        verbose (bool): Whether to print detailed information during training.
        train_loss_plot (list): A list to store training loss per epoch.
        valid_loss_plot (list): A list to store validation loss per epoch.
        train_acc_plot (list): A list to store training accuracy per epoch.
        valid_acc_plot (list): A list to store validation accuracy per epoch.
        epochs (list): A list to store the epoch numbers.
        max_acc_value (float): The highest validation accuracy achieved.
        max_acc_epoch (int): The epoch number at which the highest validation accuracy was achieved.
    """
    def __init__(self, model,hparams, optimizer = None , loss_function = None, scheduler=None, early_stop=False, verbose=False):
        self.model = model
        if optimizer is None :
            self.optimizer = optim.Adam(model.parameters(), lr = hparams.learning_rate)
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if loss_function is None : 
            self.loss_function = nn.BCEWithLogitsLoss()
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.hparams = hparams
        self.early_stop = early_stop
        self.verbose = verbose
        self.train_loss_plot = []
        self.valid_loss_plot = []
        self.train_acc_plot = []
        self.valid_acc_plot = []
        self.epochs = []
        self.max_acc_value = 0
        self.max_acc_epoch = 0
        

    def warm_up_gpu(self,train_loader):
        """
        Performs a GPU warm-up using the training loader.
        """

        for _ in trange(3, desc="Warming up GPU"):
            self.run_model(train_loader, mode='train')

    def run_model(self, loader, timer_starter = None,timer_ender = None, timings = None, mode='train'):
        """
        Runs the model for one epoch of training or evaluation.
        
        Returns:
            A tuple of (losses, accuracies) for the given data loader.
        """
        self.model.train() if mode == 'train' else self.model.eval()

        losses = []
        accuracies = []


        for data, target,id in loader:
            data_batch_shape = data.shape[0]
            data = torch.mean(data, dim=1).view(data_batch_shape, 1, data.shape[2])
            data, target = data.to(self.device), target.to(self.device)

            if timings is not None:
                timer_starter.record(torch.cuda.Stream())

                with torch.set_grad_enabled(mode == 'train'):
                    output = self.model(data)
                    loss = self.loss_function(output, target)
                    torch.cuda.synchronize()

                    timer_ender.record(torch.cuda.Stream())
                    torch.cuda.synchronize()
                    curr_time = timer_starter.elapsed_time(timer_ender)
                    timings.append(curr_time)

            if timings is None :
                with torch.set_grad_enabled(mode == 'train'):
                    output = self.model(data)
                    loss = self.loss_function(output, target)


            preds = (torch.sigmoid(output) > self.hparams.threshold) * 1
            accuracy = torch.mean((preds == target).type(torch.float))

            losses.append(loss.item())
            accuracies.append(accuracy.item())

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return losses, accuracies

    def run_epoch(self,train_loader,valid_loader,timer_starter = None, timer_ender = None, timings = None):
        """
        Runs one epoch of training and validation.
        
        Returns:
            A tuple of average training loss, average validation loss, average training accuracy, and average validation accuracy.
        """

        train_losses, train_accuracy = self.run_model(train_loader,timer_starter, timer_ender,timings, mode='train')
        valid_losses, valid_accuracy = self.run_model(valid_loader, mode='eval')

        avg_train_loss = np.mean(train_losses)
        avg_valid_loss = np.mean(valid_losses)
        avg_train_acc = np.mean(train_accuracy)
        avg_valid_acc = np.mean(valid_accuracy)

        return avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc


    # def check_early_stopping(self, valid_acc):
    #     stop = False
    #     if self.early_stop:
    #         self.scheduler.step(valid_acc)
    #         current_lr = self.optimizer.param_groups[0]['lr']
    #         stop = current_lr < self.hparams.stopping_rate
    #         if stop:
    #             print(f"Early stopping for a learning rate = {current_lr}")
    #     return stop



    def train(self,train_loader,valid_loader,timer_starter,timer_ender,timings = None,name_model = str, checkpoint_path = './model_checkpoints'):
        """
        Trains the model using the specified training and validation loaders, with optional GPU timing. It saves the model checkpoint in the specified directory path.

        Returns:
            A tuple of training and validation loss and accuracy plots.
        """

        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"Training of the model {str(self.model)} is starting ...")

        for epoch in trange(self.hparams.num_epochs, desc="Training Epochs"):
            train_loss, valid_loss, train_acc, valid_acc = self.run_epoch(train_loader,valid_loader, timer_starter= timer_starter, timer_ender=timer_ender,timings=timings)
            self.train_loss_plot.append(train_loss)
            self.valid_loss_plot.append(valid_loss)
            self.train_acc_plot.append(train_acc)
            self.valid_acc_plot.append(valid_acc)
            self.epochs.append(epoch)   

            if self.verbose:
                print(f"[Epoch {epoch + 1}/{self.hparams.num_epochs}] [Train Loss: {train_loss:.4f}] [Train Acc: {train_acc:.4f}] [Valid Loss: {valid_loss:.4f}] [Valid Acc: {valid_acc:.4f}]")

            # if self.check_early_stopping(valid_acc):
            #     break
            if (epoch + 1) % self.hparams.checkpoint_interval == 0:
                save_path = f"{checkpoint_path}/{name_model}_epoch_{epoch+1}.pth"
                torch.save(self.model, save_path)
                print(f"Model saved to {save_path}")

        if timings : 
            total_time = np.sum(timings)
        
        print(f"Total time of training is {total_time/1000} s and Time per epoch is : {total_time / ((self.epochs[-1] + 1) * 1000)} s/epoch")

        self.train_loss_plot = np.array(self.train_loss_plot)
        self.valid_loss_plot = np.array(self.valid_loss_plot)
        self.train_acc_plot = np.array(self.train_acc_plot)
        self.valid_acc_plot = np.array(self.valid_acc_plot)

        return self.train_loss_plot, self.valid_loss_plot, self.train_acc_plot, self.valid_acc_plot
    
    def best_model(self,interval_length,name_model,checkpoint_path = './model_checkpoints'):
        """
        Determines the best model based on validation accuracy over specified intervals.

        Returns:
            The model with the highest average validation accuracy within the best interval.
        """
        interval_averages = [(sum(self.valid_acc_plot[i:i+interval_length]) / interval_length) for i in range(0, self.hparams.num_epochs, interval_length)]

        # Find the interval with the highest average accuracy
        max_avg_acc_index = interval_averages.index(max(interval_averages))
        best_interval_start = max_avg_acc_index * interval_length
        best_interval_end = best_interval_start + interval_length

        filtered_accuracies = [(i + 1, acc) for i, acc in enumerate(self.valid_acc_plot) if best_interval_start < i < best_interval_end]
        filtered_accuracies_multiple = [(epoch, acc) for epoch, acc in filtered_accuracies if epoch % self.hparams.checkpoint_interval == 0]

        # Now, find the maximum accuracy and its corresponding epoch from this filtered list
        if filtered_accuracies_multiple:
            self.max_acc_epoch, self.max_acc_value = max(filtered_accuracies_multiple, key=lambda x: x[1])


        model_MLP_best = torch.load(f"{checkpoint_path}/{name_model}_epoch_{self.max_acc_epoch}.pth")

        return model_MLP_best
    
    def predict(self,model,loader,binary = True):
        """
        Makes predictions using the given model and data loader (eventually evaluation loader).

        Returns:
            A tuple of actual labels and predicted labels.
        """

        model.eval()
        model.to(self.device)

        all_preds = []
        all_true_labels = []


        
        for data, target, id in loader:
            data_batch_shape = data.shape[0]
            data = torch.mean(data, dim=1).view(data_batch_shape, 1,data.shape[2])
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                output = model(data)
                if binary :
                    preds = (torch.sigmoid(output) > self.hparams.threshold)*1
                    all_preds.extend(preds.cpu().numpy())
                    all_true_labels.extend(target.cpu().numpy())
                else :
                    all_preds.extend(output.cpu().numpy())
                    all_true_labels.extend(target.cpu().numpy())




        all_preds = np.array(all_preds)
        all_true_labels = np.array(all_true_labels)

        if all_true_labels.ndim > 1 and all_true_labels.shape[1] > 1:
            all_true_labels = np.argmax(all_true_labels, axis=1)

        # accuracy = accuracy_score(all_true_labels, all_preds)
        # conf_matrix = confusion_matrix(all_true_labels, all_preds)
        
        labels_res = all_true_labels
        preds_res = all_preds

        return labels_res, preds_res
    

import torch

def evalBuilder(data_final,model, eval_loader, device =torch.device("cuda" if torch.cuda.is_available() else "cpu"),  threshold= 0.5):
    """
    Evaluate the given model on the provided data loader, computing the predictions
    and their probabilities. Returns a dataframe containing only the evaluation set
    and the prediction results of the classifier.

    Parameters:
    - model: The machine learning model to be evaluated.
    - eval_loader: DataLoader containing the evaluation dataset.
    - device: The device (CPU or GPU) where the computations will be executed.
    - threshold: The threshold used to convert probabilities into binary predictions.

    Returns:
    - data_test: A dataframe containing the IDs, predicted results, and predicted probabilities
                 of the evaluation set.
    """
    model.eval()
    model.to(device)


    ids_eval_set = []
    dict_eval_preds = {}
    dict_eval_probs = {}

    for data, labels, ids in eval_loader:
        data_batch_shape = data.shape[0]
        data = torch.mean(data, dim=1).view(data_batch_shape, 1, data.shape[2])
        data, target = data.to(device), labels.to(device)
        
        with torch.no_grad():
            
            output = model(data)

        sigmoid_output = torch.sigmoid(output)
        preds = (sigmoid_output > threshold) * 1
        
        ids_eval_set.extend(ids.view(-1).tolist())
        batch_id_to_prediction = dict(zip(ids.view(-1).tolist(), preds.view(-1).tolist()))
        batch_id_to_probability = dict(zip(ids.view(-1).tolist(), sigmoid_output.view(-1).tolist()))

        dict_eval_preds.update(batch_id_to_prediction)
        dict_eval_probs.update(batch_id_to_probability)

    data_final['pred_result'] = data_final['id'].map(dict_eval_preds)
    data_final['pred_probability'] = data_final['id'].map(dict_eval_probs)

    data_test = data_final[data_final['id'].isin(ids_eval_set)]

    return data_test

    

# def calculate_tpr_fpr(clf,model_list, eval_loader, device):
#     tpr_list = []
#     fpr_list = []

#     for model in model_list:
#         all_true_labels, all_scores = clf.predict(model=model, loader=eval_loader, binary = False )
#         fpr, tpr, thresholds = roc_curve(all_true_labels, all_scores)
#         tpr_list.append(np.interp(mean_fpr, fpr, tpr))
#         tpr_list[-1][0] = 0.0

#     mean_tpr = np.mean(tpr_list, axis=0)
#     mean_tpr[-1] = 1.0
#     std_tpr = np.std(tpr_list, axis=0)

#     return mean_fpr, mean_tpr, std_tpr
    





    




    





















