"""
Evaluate Attack Ability - Test Your Attack Against Reference Models
Author: Lingfang Li
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

################################################################################################
## Define your attack here (same as in Competition.py)
################################################################################################

def your_attack(model, X, y, device):
    X_adv = Variable(X.data)
    """
    Replace this with your attack implementation from Competition.py
    Example: copy your adv_attack function here
    """
    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    X_adv = Variable(X_adv.data + random_noise)

    return X_adv

################################################################################################
## End of attack definition
################################################################################################

def eval_adv_test(model, device, test_loader, attack_method):
    model.eval()
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)

        # Generate adversarial examples (needs gradients)
        adv_data = attack_method(model, data, target, device=device)

        # Evaluate on adversarial examples (no gradients needed)
        with torch.no_grad():
            output = model(adv_data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct / len(test_loader.dataset)

def main():

    ################################################################################################
    ## Step 1: Download reference models
    ## Go to Canvas → COMP219 Module → Lecture 14 → Download "Defenders.zip"
    ## Extract the zip file, then specify the path below
    ################################################################################################

    defenders_path = ".\Defenders"  # change this to your extracted Defenders folder path

    ################################################################################################
    ## Step 2: Your attack function is defined above (see your_attack function)
    ################################################################################################

    attack_to_test = your_attack  # use the attack defined above

    ################################################################################################
    ## End of configuration
    ################################################################################################

    parser = argparse.ArgumentParser(description='Attack Evaluation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    args = parser.parse_args(args=[])

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    print("="*80)
    print("COMP219 Assignment - Attack Evaluation")
    print("="*80)
    print(f"Defenders path: {defenders_path}")
    print(f"Device: {device}")
    print("="*80)

    # check if defenders path exists
    if not os.path.exists(defenders_path):
        print(f"\nERROR: Defenders folder not found at '{defenders_path}'")
        print("\nPlease:")
        print("1. Go to Canvas → COMP219 Module → Lecture 14")
        print("2. Download 'Defenders.zip'")
        print("3. Extract the zip file")
        print("4. Update 'defenders_path' variable in this script")
        return

    # find all .pt files in defenders folder
    model_files = [f for f in os.listdir(defenders_path) if f.endswith('.pt')]
    model_files.sort()

    if len(model_files) == 0:
        print(f"\nERROR: No .pt model files found in '{defenders_path}'")
        print("Please check the folder contains the extracted model files.")
        return

    print(f"Found {len(model_files)} reference models")
    print("="*80)

    # load test data
    test_set = torchvision.datasets.FashionMNIST(
        root='data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # build attack matrix: [1 attack] x [N models]
    num_models = len(model_files)
    attack_matrix = np.zeros(num_models)

    print("\nEvaluating your attack...")
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(defenders_path, model_file)
        print(f"  [{i+1}/{num_models}] {model_file}...", end=" ", flush=True)

        # load model
        model = Net().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # evaluate attack
        robust_acc = eval_adv_test(model, device, test_loader, attack_to_test)
        attack_matrix[i] = robust_acc
        print(f"{robust_acc*100:.2f}%")

        del model

    # calculate attack score (mean of 1/robust_acc)
    attack_scores = 1.0 / (attack_matrix + 1e-10)  # avoid division by zero
    attack_score = np.mean(attack_scores)

    # display results
    print("\n" + "="*80)
    print("ATTACK MATRIX (Your Attack vs Reference Models)")
    print("="*80)
    print(f"\n{'Model':<20} {'Robust Accuracy':<20} {'Attack Score (1/acc)'}")
    print("-"*80)
    for i, model_file in enumerate(model_files):
        print(f"{model_file:<20} {attack_matrix[i]*100:>6.2f}%              {attack_scores[i]:>6.2f}")
    print("-"*80)
    print(f"{'Mean':<20} {np.mean(attack_matrix)*100:>6.2f}%              {attack_score:>6.2f}")
    print("="*80)

    print("\n" + "="*80)
    print("ATTACK SCORE")
    print("="*80)
    print(f"Your Attack Score: {attack_score:.4f}")
    print("="*80)
    print("\nNote:")
    print("  - Higher score is better (stronger attack)")
    print("  - Attack Score = mean(1/robust_accuracy)")
    print("  - Lower robust accuracy → stronger attack → higher score")
    print("  - Final grading will normalize scores across ALL students")
    print("  - This self-test uses reference models only")

if __name__ == '__main__':
    main()
