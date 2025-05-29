import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import random
import csv
from tqdm import tqdm
from itertools import product
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_model(pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model

def train_model(model, optimizer, scheduler,
                train_loader, val_loader, test_loader,
                num_epochs, model_name='finetuned'):
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter(f'runs/{model_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        scheduler.step()

        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc  = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)


        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')

        print(f"{model_name} Epoch {epoch}/{num_epochs} "
              f"Train Loss: {epoch_train_loss:.4f} "
              f"Val Loss: {epoch_val_loss:.4f} "
              f"Val Acc: {epoch_val_acc:.4f}")

    model.load_state_dict(torch.load(f'best_{model_name}.pth'))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"{model_name} Final Test Acc: {test_acc:.4f}")

    return train_losses, val_losses, val_accs, test_acc

def main():
    set_seed(42)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = './caltech-101/caltech-101/101_ObjectCategories'
    base_ds = datasets.ImageFolder(data_dir)
    bg = base_ds.class_to_idx.get('BACKGROUND_Google')
    if bg is not None:
        base_ds.samples = [s for s in base_ds.samples if s[1] != bg]
        classes = sorted({os.path.basename(os.path.dirname(s[0])) for s in base_ds.samples})
        base_ds.class_to_idx = {c:i for i,c in enumerate(classes)}
        base_ds.classes = classes
    assert len(base_ds.classes) == 101


    targets = np.array([s[1] for s in base_ds.samples])
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(targets):
        idxs = np.where(targets==cls)[0]
        np.random.shuffle(idxs)
        n = len(idxs)
        train_n = min(30, n)
        val_n = 6
        train_idx += idxs[:train_n-val_n].tolist()
        val_idx   += idxs[train_n-val_n:train_n].tolist()
        test_idx  += idxs[train_n:].tolist()

    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds   = Subset(datasets.ImageFolder(data_dir, transform=eval_transform),  val_idx)
    test_ds  = Subset(datasets.ImageFolder(data_dir, transform=eval_transform),  test_idx)

    loaders = {
        'train': DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2),
        'val':   DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2),
        'test':  DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)
    }
    train_loader = loaders['train']
    val_loader   = loaders['val']
    test_loader  = loaders['test']


    epochs_list  = [60, 100]
    lr_fc_list   = [1e-3, 5e-4]
    lr_base_list = [1e-4, 1e-5]


    all_results = [] 
    with open('grid_search.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epochs', 'lr_fc', 'lr_base', 'val_acc', 'test_acc'])

        for epochs, lr_fc, lr_base in product(epochs_list, lr_fc_list, lr_base_list):
            label = f"epoch{epochs}_fclr{lr_fc}_baselr{lr_base}"
            print(f"==> Training {label}")
            model = build_model(pretrained=True)
            optimizer = optim.Adam([
                {'params': [p for n,p in model.named_parameters() if not n.startswith('fc')], 'lr': lr_base},
                {'params': model.fc.parameters(), 'lr': lr_fc}
            ])
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

            t_losses, v_losses, v_accs, test_acc = train_model(
                model, optimizer, scheduler,
                train_loader, val_loader, test_loader,
                epochs, f"finetuned_{label}"
            )

            writer.writerow([epochs, lr_fc, lr_base, round(v_accs[-1], 4), round(test_acc, 4)])

            all_results.append((label, t_losses, v_losses, v_accs))
            

    os.makedirs('plots', exist_ok=True)
    # 1. 训练损失
    plt.figure()
    for label, t_losses, *_ in all_results:
        plt.plot(range(1, len(t_losses)+1), t_losses, label=label)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/train_loss.png')
    plt.close()

    # 2. 验证损失
    plt.figure()
    for label, _, v_losses, _ in all_results:
        plt.plot(range(1, len(v_losses)+1), v_losses, label=label)
    plt.title('Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/val_loss.png')
    plt.close()

    # 3. 验证准确率
    plt.figure()
    for label, *_, v_accs in all_results:
        plt.plot(range(1, len(v_accs)+1), v_accs, label=label)
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/val_accuracy.png')
    plt.close()

    print("\n==> Training from scratch... (no grid) ")
    model_scratch = build_model(pretrained=False)
    opt_s = optim.Adam(model_scratch.parameters(), lr=1e-3)
    sch_s = StepLR(opt_s, step_size=10, gamma=0.1)
    train_model(model_scratch, opt_s, sch_s,
                loaders['train'], loaders['val'], loaders['test'],
                60, 'scratch')

if __name__ == '__main__':
    main()
