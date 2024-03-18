from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import torch

def DataSplitLoader(data_final, hparams, embedding_type, eval_size = 0.2, train_size = 0.7, valid_size = 0.1, random_state = 42):
    """
    Prepares the training, evaluation, and validation loaders.
    
    Parameters:
    - data_final: A DataFrame containing 'embedding' and 'label' columns.
    - hparams: An instance of HParams class containing hyperparameters.
    - embedding_type : A string referring to the embedding type of the dataset.
    - eval/train/valid_size : size for the specific set.
    
    Returns:
    - train_loader: DataLoader for the training dataset.
    - eval_loader: DataLoader for the evaluation dataset.
    - valid_loader: DataLoader for the valid dataset.
    """
    # Convert DataFrame columns to numpy arrays
    embeddings = np.array(data_final[data_final['embedding_type']== embedding_type ]['embedding'].tolist())
    labels = (data_final[data_final['embedding_type']== embedding_type ]['label']).values
    ids= data_final[data_final['embedding_type']== embedding_type]['id'].values

    

    # Split the data into training, validation, and test sets
    X_temp, X_valid, y_temp, y_valid, id_temp, id_valid = train_test_split(embeddings, labels, ids, test_size=valid_size, random_state=42)
    eval_size_adjusted = eval_size / (train_size + eval_size)
    
    X_train, X_eval, y_train, y_eval, id_train, id_eval = train_test_split(X_temp, y_temp, id_temp, test_size = eval_size_adjusted, random_state = random_state)

    # Convert numpy arrays to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).view(-1,1)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1,1)

    id_train_tensor = torch.tensor(id_train, dtype=torch.float32).view(-1,1)
    id_eval_tensor = torch.tensor(id_eval, dtype=torch.float32).view(-1,1)
    id_valid_tensor = torch.tensor(id_valid, dtype=torch.float32).view(-1,1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor,id_train_tensor)
    eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor,id_eval_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor,id_valid_tensor)

    # Create DataLoaders
    batch_size = hparams.batch_size
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / class_sample_count
    samples_weights = np.array([weights[t] for t in y_train])

    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True) # type: ignore

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, valid_loader
