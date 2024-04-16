from dataset import MNIST
from model import LeNet5, CustomMLP, LeNet5_Imp
from torchvision.transforms import RandomRotation
from tqdm import tqdm_notebook
import torch

def train(model, trn_loader, device, criterion, optimizer, augmentation=0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    random_rotation = RandomRotation(degrees=(-30, 30), fill=(-0.4242))

    for images, label in trn_loader:
        images, label = images.to(device), label.to(device)

        if augmentation == 0:
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        else:
            # data Augmentation
            for _ in range(augmentation):
                image = random_rotation(images)
                optimizer.zero_grad()
                outputs = model(image)

                loss = criterion(outputs, label)
                loss.backward()

                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

    acc = 100 * correct / total
    trn_loss = total_loss / len(trn_loader)

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()  
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for image, targets in tst_loader:
            image, targets = image.to(device), targets.to(device)
            outputs = model(image)
            loss = criterion(outputs, targets)
            total_loss += loss.item() 

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total

    tst_loss = total_loss / len(tst_loader)
    return tst_loss, acc

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Epoch = 20
    batch_size = 32

    trainset = MNIST('../data/train')
    testset = MNIST('../data/test')
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    # LeNet5
    model = LeNet5().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    train_acces = []
    
    test_losses = []
    test_acces = []
    
    print('LeNet5')
    print(f'Parameter Num: {sum(p.numel() for p in model.parameters())}')
    for epoch in tqdm_notebook(range(Epoch)):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, device, criterion)
        print(f'Epoch {epoch+1}\tTrain loss: {train_loss:.04f}\tTrain Acc: {train_acc:.04f}')
        print(f'Epoch {epoch+1}\tTest loss: {test_loss:.04f}\tTest Acc: {test_acc:.04f}')
    
        print()
    
        train_losses.append(train_loss)
        train_acces.append(train_acc)
    
        test_losses.append(test_loss)
        test_acces.append(test_acc)
    
    np.save('../result/LeNet5_train_loss', train_losses)
    np.save('../result/LeNet5_train_acc', train_acces)
    np.save('../result/LeNet5_test_loss', test_losses)
    np.save('../result/LeNet5_test_acc', test_acces)
    
    # CustomMLP
    model = CustomMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    train_acces = []
    
    test_losses = []
    test_acces = []
    
    print('CustomMLP')
    print(f'Parameter Num: {sum(p.numel() for p in model.parameters())}')
    for epoch in tqdm_notebook(range(Epoch)):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, device, criterion)
        print(f'Epoch {epoch+1}\tTrain loss: {train_loss:.04f}\tTrain Acc: {train_acc:.04f}')
        print(f'Epoch {epoch+1}\tTest loss: {test_loss:.04f}\tTest Acc: {test_acc:.04f}')
    
        print()
    
        train_losses.append(train_loss)
        train_acces.append(train_acc)
    
        test_losses.append(test_loss)
        test_acces.append(test_acc)
    
    np.save('../result/CustomMLP_train_loss', train_losses)
    np.save('../result/CustomMLP_train_acc', train_acces)
    np.save('../result/CustomMLP_test_loss', test_losses)
    np.save('../result/CustomMLP_test_acc', test_acces)
    
    # Improved LeNet5
    model = LeNet5_Imp().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    train_acces = []
    
    test_losses = []
    test_acces = []
    
    print('Improved LeNet5')
    for epoch in tqdm_notebook(range(Epoch)):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, device, criterion)
        print(f'Epoch {epoch+1}\tTrain loss: {train_loss:.04f}\tTrain Acc: {train_acc:.04f}')
        print(f'Epoch {epoch+1}\tTest loss: {test_loss:.04f}\tTest Acc: {test_acc:.04f}')
    
        print()
    
        train_losses.append(train_loss)
        train_acces.append(train_acc)
    
        test_losses.append(test_loss)
        test_acces.append(test_acc)
    
    np.save('../result/LeNet5_Imp_train_loss', train_losses)
    np.save('../result/LeNet5_Imp_train_acc', train_acces)
    np.save('../result/LeNet5_Imp_test_loss', test_losses)
    np.save('../result/LeNet5_Imp_test_acc', test_acces)
    
    # Improved LeNet5 with Augmentation
    model = LeNet5_Imp().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    train_acces = []
    
    test_losses = []
    test_acces = []
    
    print('Improved LeNet5 with Augmentation')
    for epoch in tqdm_notebook(range(Epoch)):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer, augmentation=10)
        test_loss, test_acc = test(model, test_loader, device, criterion)
        print(f'Epoch {epoch+1}\tTrain loss: {train_loss:.04f}\tTrain Acc: {train_acc:.04f}')
        print(f'Epoch {epoch+1}\tTest loss: {test_loss:.04f}\tTest Acc: {test_acc:.04f}')
    
        print()
    
        train_losses.append(train_loss)
        train_acces.append(train_acc)
    
        test_losses.append(test_loss)
        test_acces.append(test_acc)
    
    np.save('../result/LeNet5_Imp_Augment_train_loss', train_losses)
    np.save('../result/LeNet5_Imp_Augment_train_acc', train_acces)
    np.save('../result/LeNet5_Imp_Augment_test_loss', test_losses)
    np.save('../result/LeNet5_Imp_Augment_test_acc', test_acces)

if __name__ == '__main__':
    main()
