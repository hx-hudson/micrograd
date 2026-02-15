import torchvision
import torchvision.transforms as transforms
import numpy as np
from model import CNN
from engine import cross_entropy, Optimizer

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

def normalize(X):
    return (X - MNIST_MEAN) / MNIST_STD

def evaluate(model, X_test, y_test, batch_size=128):
    correct = 0
    total = 0

    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        logits = model.forward(X_batch)
        preds = np.argmax(logits.data, axis=1)

        correct += np.sum(preds == y_batch)
        total += len(y_batch)

    return correct / total

def get_batches(X, y, batch_size):
    indices = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def train(X_train, y_train, batch_size, model, optimizer, X_test, y_test, lr=0.01, epochs=2):
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            optimizer.zero_grad()

            logits = model.forward(X_batch)
            loss = cross_entropy(logits, y_batch)

            loss.backward_all()
            optimizer.update(lr)

            total_loss += loss.data
            num_batches += 1

        avg_loss = total_loss / num_batches
        test_acc = evaluate(model, X_test, y_test)
        print(f"epoch {epoch:4d} | loss {avg_loss:.4f} | test acc {test_acc:.4f}")

if __name__ == "__main__":
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    X_train = np.array([img.numpy() for img, _ in train_dataset])  # (60000, 1, 28, 28)
    y_train = np.array([label for _, label in train_dataset])       # (60000,)

    X_test = np.array([img.numpy() for img, _ in test_dataset])    # (10000, 1, 28, 28)
    y_test = np.array([label for _, label in test_dataset])         # (10000,)

    X_train = normalize(X_train)
    X_test  = normalize(X_test)

    model = CNN()
    optimizer = Optimizer(model.parameters())

    train(
        X_train, y_train,
        batch_size=64,
        model=model,
        optimizer=optimizer,
        X_test=X_test,
        y_test=y_test,
        lr=0.01,
        epochs=10,
    )