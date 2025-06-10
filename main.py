# Demo: Annabelle's research as described in the corresponding paper

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from independent_validation import *
import seaborn as sns
import os


# initiate some values
np.random.seed(42)

folder = 'plots'
os.makedirs(f'{folder}', exist_ok=True)

sns.set_style("ticks")  # alternatives: 'white', 'dark', 'darkgrid', etc.
sns.set_context("paper", font_scale=1.5)
plt.rc('font', family='sans-serif')

# Define the x-axis values (from 0 to 1)
x_axis = np.linspace(0.0, 1, 10000)

wine = load_wine()
X, y = wine.data, wine.target


def main():
    '''
    Part 1: Is there a group difference?
        - test if SVM is better than chance
        - also test binary case
    Part 2: BACC
        - Compare classifiers balanced accuracy
        - usage of prob_greater_than()
    Part 3: ACC
        - Compare classifiers accuracy
        - usage of prob_greater_than
    Part 4: Development
        - Development of ACC over trainset size
    '''
    group_difference()
    compare_clfs()
    print('\nNow compare global accuracy:')
    compare_clfs(mode='acc')
    development()


def group_difference():
    # Part 1 Group Difference

    # use Independent validation
    iv_svm = IV(X, y, SVC(gamma='scale'))  # initiating
    iv_svm.run_iv(start_trainset_size=5)  # classify samples
    iv_svm.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)  # use mcmc to compute posterior
    bacc_svm_dist = iv_svm.get_bacc_dist()  # get desired output
    print("Mode (MAP) value:", bacc_svm_dist.map())


    # 95% CI
    lower_bound = bacc_svm_dist.ppf(0.025)
    upper_bound = bacc_svm_dist.ppf(0.975)
    print("95% CI:", lower_bound, "-", upper_bound)

    # calculate below 1/3
    density_at_one_third = bacc_svm_dist.cdf(1/3)  # this is density up to one third
    print("Density up to 1/3:", density_at_one_third)

    fig, ax = plt.subplots(figsize=(8, 6))

    pdf = bacc_svm_dist.pdf(x_axis)
    ax.plot(x_axis, pdf, label="Support Vector Machine", color="blue", lw=2)
    ax.fill_between(x_axis, pdf, color="blue", alpha=0.3)

    # Set axis labels and title
    ax.set_xlabel('BACC', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('SVM Distribution of BACC Score', fontsize=16)

    # Add the legend
    ax.legend(title="Classifier", fontsize=12, title_fontsize=12, loc="best")
    plt.savefig(f'{folder}/figure1')
    plt.show()


    # Binary case:
    selected_indices = np.where(y != 2)[0]  # only Barolo and Lugana

    X_bin = X[selected_indices]
    y_bin = y[selected_indices]

    iv_svm = IV(X_bin, y_bin, SVC(gamma='scale'))

    iv_svm.run_iv(start_trainset_size=5)
    iv_svm.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)

    # get sensitivity
    barolo = iv_svm.get(key=0)
    print("Barolo (MAP) value:", barolo.map())

    # get specificity
    lugana = iv_svm.get(key=1)
    print("Lugana (MAP) value:", lugana.map())



def compare_clfs(mode="bacc"):
    assert mode in ['bacc', 'acc']
    bacc = mode == 'bacc'
    acc = mode == 'acc'

    # Part 2 BACC and 3 ACC

    # Define the classifiers.
    classifiers = {
        'Logistic_Regression': LogisticRegression(solver='newton-cg'),
        'Support_Vector_Machine': SVC(gamma='scale'),
        'K_Nearest_Neighbors': KNeighborsClassifier(),
        'Random_Forest': RandomForestClassifier()
    }

    # Dictionary to store the accuracy distributions.
    dists = {}
    for clf_name, clf in classifiers.items():
        print(f"\nRunning IV for classifier: {clf_name}")

        iv_instance = IV(X, y, clf)
        iv_instance.run_iv(start_trainset_size=5)
        iv_instance.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)

        if bacc:
            dist_clf = iv_instance.get_bacc_dist(plot=False)
        if acc:
            dist_clf = iv_instance.get_acc_dist(plot=False)
        dists[clf_name] = dist_clf

        print(f'{clf_name} MAP value...', dist_clf.map())

    # compare Random Forest and Logistic Regression
    prob_RF_gt_LR = dists['Random_Forest'].is_greater_than(dists['Logistic_Regression'])
    print('Probability that RF accuracy is greater than LR accuracy:', prob_RF_gt_LR)

    # set colours for classifiers
    colors = {
        'Support_Vector_Machine': 'tab:blue',
        'K_Nearest_Neighbors': 'tab:orange',
        'Random_Forest': 'tab:green',
        'Logistic_Regression': 'tab:red'
    }


    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each distribution with a filled area
    for classifier, distribution in dists.items():
        pdf = distribution.pdf(x_axis)
        ax.plot(x_axis, pdf, label=classifier, color=colors[classifier], lw=2)
        ax.fill_between(x_axis, pdf, color=colors[classifier], alpha=0.3)

    # Set axis labels and title
    if bacc:
        ax.set_xlabel('BACC', fontsize=14)
        ax.set_title('Distributions of BACC Scores', fontsize=16)
    if acc:
        ax.set_xlabel('ACC', fontsize=14)
        ax.set_title('Distributions of ACC Scores', fontsize=16)

    # Add the legend
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(title="Classifier", fontsize=12, title_fontsize=12, loc="best")

    plt.tight_layout()
    if bacc:
        plt.savefig(f'{folder}/figure2.png')
    if acc:
        plt.savefig(f'{folder}/figure3.png')
    plt.show()

# part 3 ACC

def development():
    print("\n=== PART 4: Development curve (Balanced Accuracy vs. Training Set Size) ===")
    max_trainset_size = 100

    iv_dev = IV(X, y, RandomForestClassifier())
    iv_dev.run_iv(start_trainset_size=5)
    iv_dev.compute_posterior(burn_in=1000, thin=10, step_size=0.1, num_samples=1000)
    trainset_sizes = np.arange(1, max_trainset_size + 1)

    mean_acc_list, lower_bound_list, upper_bound_list = iv_dev.get_development(key='acc', n=101, plot=False,
                                                                               confidence_range=0.5)

    # Plot the development curve.
    plt.figure(figsize=(10, 6))
    plt.plot(trainset_sizes, mean_acc_list, label='Mean Accuracy', color='blue')
    plt.fill_between(trainset_sizes, lower_bound_list, upper_bound_list, color='gray', alpha=0.3,
                     label='50% Credible Interval')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Development Curve: Accuracy vs. Training Set Size (SVM)")
    plt.legend()
    plt.savefig(f'{folder}/figure4.png')
    plt.show()



if __name__ == "__main__":
    main()