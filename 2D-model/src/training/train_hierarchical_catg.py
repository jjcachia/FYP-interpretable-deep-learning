# TODO: Complete the implementation of the training and testing steps for the hierarchical model with categorical tasks and final prediction.

import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

def _train_or_test(model, data_loader, optimizer, device, is_train=True):
    """Perform training or testing steps on given model and data loader."""
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_correct = [0] * 5  # Assuming 5 tasks
    total_samples = [0] * 5  # Assuming 5 tasks
    
    
    final_pred_targets = [[] for _ in range(5)]
    final_pred_outputs = [[] for _ in range(5)]
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars,_,_ in tqdm(data_loader, leave=False):  # Assuming targets is a list of target tensors for each task
            X = X.to(device)
            bweights_chars = [b.float().unsqueeze(1).to(device) for b in bweights_chars]
            outputs = model(X)
            # print("Weights shape:", bweights_chars.shape)
            # print("Output shape:", [o.shape for o in outputs])
            # print("Target shape:", [o.shape for o in targets])
            loss = 0
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # print("Output shape:", output.shape)
                # print("Target shape:", target.shape)
                # print("Weight shape:", bweights_chars[i].shape)
                target = target.to(device)
                task_loss = (torch.nn.functional.cross_entropy(output, target, reduction="none")*bweights_chars[i]).mean()
                loss += task_loss
                # print("Task loss:", task_loss)

                # Compute accuracy for each task
                _, preds = torch.max(output, 1)
                _, tars = torch.max(target, 1)
                # print("Preds shape:", preds.shape)
                # print("Tars shape:", tars.shape)
                total_correct[i] += (preds == tars).sum().item()
                total_samples[i] += target.size(0)

                # Collect data for balanced accuracy
                final_pred_targets[i].extend(tars.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())
            
            total_loss += loss.item() / len(outputs)  # Average the loss across tasks
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    average_loss = total_loss / len(data_loader)
    accuracies = [correct / samples for correct, samples in zip(total_correct, total_samples)]
    balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    return average_loss, accuracies, balanced_accuracies


def train_step(model, data_loader, optimizer, device):
    """Train the model for one epoch."""
    train_loss, train_accuracies, train_balanced_accuracies = _train_or_test(
        model, data_loader, optimizer, device, is_train=True
    )
    print(f"Train loss: {train_loss:.5f}")
    for i, (acc, bal_acc) in enumerate(zip(train_accuracies, train_balanced_accuracies), 1):
        print(f"Task {i} - Train Accuracy: {acc*100:.2f}%, Train Balanced Accuracy: {bal_acc*100:.2f}%")

def test_step(model, data_loader, device):
    """Evaluate the model."""
    test_loss, test_accuracies, test_balanced_accuracies = _train_or_test(
        model, data_loader, None, device, is_train=False
    )
    print(f"Test loss: {test_loss:.5f}")
    for i, (acc, bal_acc) in enumerate(zip(test_accuracies, test_balanced_accuracies), 1):
        print(f"Task {i} - Test Accuracy: {acc*100:.2f}%, Test Balanced Accuracy: {bal_acc*100:.2f}%")