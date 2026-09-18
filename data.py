from sklearn.datasets import make_moons
import torch
# from utils import plot_dataset
from sklearn.model_selection import train_test_split
X,y =make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42

)
X_train,X_test, y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)
# print(f"{X_train.shape}\n {y_train.shape}\n {X_test.shape}\n {y_test.shape}")

# plot_dataset(X,y)