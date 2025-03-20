import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
from independent_validation import *
import seaborn as sns
# import services as sv

np.random.seed(42)



# PART 0. Create a balanced binary dataset.
#
# The research scenario requires a balanced dataset where random guessing yields 50%
# balanced accuracy. We use the Wine dataset and select only classes 0 and 1.
def get_wine(firstclass=30, secondclass=50, thirdclass=40):
    wine = load_wine()
    X, y = wine.data, wine.target

    indices_class0 = np.where(y == 0)[0] # barolo
    indices_class1 = np.where(y == 1)[0] # lugana
    indices_class2 = np.where(y == 2)[0] # primitivo

    # slightly unbalanced
    selected_class0 = np.random.choice(indices_class0, size=firstclass, replace=False)
    selected_class1 = np.random.choice(indices_class1, size=secondclass, replace=False)
    selected_class2 = np.random.choice(indices_class2, size=thirdclass, replace=False)
    selected_indices = np.concatenate([selected_class0, selected_class1, selected_class2])

    # Create the slightly unbalanced dataset.
    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]
    return X_balanced,y_balanced

X_balanced, y_balanced= get_wine()

#### testing for optimum
wine = load_wine()
X, y = wine.data, wine.target

iv_svm = IV(X, y, SVC(gamma='scale'))

# Compute the posterior distribution.
iv_svm.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)

bacc_svm_dist=iv_svm.get_bacc_dist()


# Set a different style (e.g., 'ticks' instead of 'whitegrid')
sns.set_style("ticks")  # alternatives: 'white', 'dark', 'darkgrid', etc.
sns.set_context("paper", font_scale=1.5)
plt.rc('font', family='sans-serif')

# Define the x-axis values (from 0 to 1)
x = np.linspace(0.4, 1, 1000)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

pdf = bacc_svm_dist.pdf(x)
ax.plot(x, pdf, label="Support Vector Machine", color="blue", lw=2)
ax.fill_between(x, pdf, color="blue", alpha=0.3)

# Set axis labels and title
ax.set_xlabel('BACC', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('SVM Distribution of BACC Score', fontsize=16)

# Add the legend
ax.legend(title="Classifier", fontsize=12, title_fontsize=12, loc="best")
plt.savefig('plots/figure1.png')
plt.show()


def part1():
    print('Part 1')
    # ---------------------------
    # PART 1. Test for a group difference.
    #
    # Annabelle’s research question: “What is the probability that the classifier’s
    # asymptotic (infinite‐data) accuracy is at least as high as random guessing (50%)?”
    #
    # We use a Support Vector Machine (SVC) for this test.

    alpha = 0.05  # significance level
    # get wine data
    X_balanced, y_balanced= get_wine()
    print("\n=== PART 1: Testing for group difference using SVM ===")
    iv_svm = IV(X_balanced, y_balanced, SVC(gamma='scale'))

    # Compute the posterior distribution.
    iv_svm.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)

    # Run IV to simulate the classifier’s performance.
    iv_svm.run_iv(start_trainset_size=5)
    # Get the estimated distribution of the overall (balanced) accuracy.
    bacc_dist = iv_svm.get_bacc_dist()
    # plt.title("Asymptotic Accuracy Distribution (SVM)")

    hist, bin_edges = bacc_dist._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]
    print("Mode (MAP) value:", mode_value)

    # 95% CI
    lower_bound = bacc_dist.ppf(0.025)
    upper_bound = bacc_dist.ppf(0.975)
    print("95% CI:", lower_bound, "-", upper_bound)

    # calculate below 1/3
    density_at_one_third = bacc_dist.cdf(1/3)  # this is density up to one third
    print("Density up to 1/3:", density_at_one_third)

    # PART 1.1 Test for a 2 group difference.

    np.random.seed(42)

    wine = load_wine()
    X, y = wine.data, wine.target

    indices_class0 = np.where(y == 0)[0] # barolo
    indices_class1 = np.where(y == 1)[0] # lugana

    # slightly unbalanced
    selected_class0 = np.random.choice(indices_class0, size=30, replace=False)
    selected_class1 = np.random.choice(indices_class1, size=50, replace=False)
    selected_indices = np.concatenate([selected_class0, selected_class1])

    # Create the slightly unbalanced dataset.
    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]

    print("\n=== PART 1.1: Testing for 2 group difference using SVM ===")
    iv_svm = IV(X_balanced, y_balanced, SVC(gamma='scale'))

    # Compute the posterior distribution.
    iv_svm.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)

    barolo=iv_svm.get_label_accuracy(0)
    hist, bin_edges = barolo._histogram
    # calculate MAP
    barolo_max = np.argmax(hist)
    barolo_mode_value = bin_edges[barolo_max]
    print("Barolo (MAP) value:", barolo_mode_value)

    lugana=iv_svm.get_label_accuracy(1)
    hist, bin_edges = lugana._histogram
    # calculate MAP
    lugana_max = np.argmax(hist)
    lugana_mode_value = bin_edges[lugana_max]
    print("Lugana (MAP) value:", lugana_mode_value)


def part2():
    print('Part 2')
    # ---------------------------
    # PART 2. Comparing classifiers
    #
    X_balanced, y_balanced= get_wine()

    # Define the classifiers.
    classifiers = {
        'Support Vector Machine': SVC(gamma='scale'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression()
    }

    # Dictionary to store the accuracy distributions.
    bacc_distributions = {}

    for clf_name, clf in classifiers.items():
        print(f"\nRunning IV for classifier: {clf_name}")

        # For SVM, reuse the existing IV (iv_svm) and recompute for trainset size 25.
        iv_instance = IV(X, y, clf)
        iv_instance.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)
        iv_instance.run_iv(start_trainset_size=5)
        acc_dist_clf = iv_instance.get_bacc_dist(plot=False)
        bacc_distributions[clf_name] = acc_dist_clf


    hist, bin_edges = bacc_distributions['Support Vector Machine']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]
    print('SVM mode value...', mode_value)

    hist, bin_edges = bacc_distributions['K-Nearest Neighbors']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]
    print('KNN mode value...', mode_value)


    hist, bin_edges = bacc_distributions['Random Forest']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]
    print('RF mode value...', mode_value)


    hist, bin_edges = bacc_distributions['Logistic Regression']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]
    print('LR mode value...', mode_value)

    prob_RF_gt_LR = bacc_distributions['Random Forest'].is_greater_than(bacc_distributions['Logistic Regression'])
    print('Probability that RF accuracy is greater than LR accuracy:', prob_RF_gt_LR)

    # Definiere Farben für die einzelnen Klassifizierer
    colors = {
        'Support Vector Machine': 'tab:blue',
        'K-Nearest Neighbors': 'tab:orange',
        'Random Forest': 'tab:green',
        'Logistic Regression': 'tab:red'
    }

    # Set a different style (e.g., 'ticks' instead of 'whitegrid')
    sns.set_style("ticks")  # alternatives: 'white', 'dark', 'darkgrid', etc.
    sns.set_context("paper", font_scale=1.5)
    plt.rc('font', family='sans-serif')

    # Define the x-axis values (from 0 to 1)
    x = np.linspace(0.4, 1, 1000)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each distribution with a filled area
    for classifier, distribution in bacc_distributions.items():
        pdf = distribution.pdf(x)
        ax.plot(x, pdf, label=classifier, color=colors[classifier], lw=2)
        ax.fill_between(x, pdf, color=colors[classifier], alpha=0.3)

    # Set axis labels and title
    ax.set_xlabel('BACC', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Distributions of BACC Scores', fontsize=16)

    # Add the legend
    ax.legend(title="Classifier", fontsize=12, title_fontsize=12, loc="best")

    plt.tight_layout()
    plt.savefig('plots/figure2.png')
    plt.show()


def part3():
    # dist_greater_than(bacc_distributions['Random Forest'],bacc_distributions['Logistic Regression'])
    print('Part 3')
    # ---------------------------
    # PART 3. Development: now with acc
    #

    X_balanced, y_balanced= get_wine()

    # Define the classifiers.
    classifiers = {
        'Support Vector Machine': SVC(gamma='scale'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression()
    }

    # Dictionary to store the accuracy distributions.
    acc_distributions = {}

    for clf_name, clf in classifiers.items():
        print(f"\nRunning IV for classifier: {clf_name}")

        # For SVM, reuse the existing IV (iv_svm) and recompute for trainset size 25.
        iv_instance = IV(X, y, clf)
        iv_instance.compute_posterior(burn_in=1500, thin=10, step_size=0.2, num_samples=1000)
        iv_instance.run_iv(start_trainset_size=5)
        acc_dist_clf = iv_instance.get_acc_dist(plot=False)
        acc_distributions[clf_name] = acc_dist_clf


    hist, bin_edges = acc_distributions['Support Vector Machine']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]

    hist, bin_edges = acc_distributions['K-Nearest Neighbors']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]

    hist, bin_edges = acc_distributions['Random Forest']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]

    hist, bin_edges = acc_distributions['Logistic Regression']._histogram
    # calculate MAP
    idx_max = np.argmax(hist)
    mode_value = bin_edges[idx_max]

    # Definiere Farben für die einzelnen Klassifizierer
    colors = {
        'Support Vector Machine': 'tab:blue',
        'K-Nearest Neighbors': 'tab:orange',
        'Random Forest': 'tab:green',
        'Logistic Regression': 'tab:red'
    }

    # Set a different style (e.g., 'ticks' instead of 'whitegrid')
    sns.set_style("ticks")  # alternatives: 'white', 'dark', 'darkgrid', etc.
    sns.set_context("paper", font_scale=1.5)
    plt.rc('font', family='sans-serif')

    # Define the x-axis values (from 0 to 1)
    x = np.linspace(0.4, 1, 1000)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each distribution with a filled area
    for classifier, distribution in acc_distributions.items():
        pdf = distribution.pdf(x)
        ax.plot(x, pdf, label=classifier, color=colors[classifier], lw=2)
        ax.fill_between(x, pdf, color=colors[classifier], alpha=0.3)

    # Set axis labels and title
    ax.set_xlabel('ACC', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Distributions of ACC Scores', fontsize=16)

    # Add the legend
    ax.legend(title="Classifier", fontsize=12, title_fontsize=12, loc="best")

    plt.tight_layout()
    plt.savefig('plots/figure3.png')
    plt.show()

    # dist_greater_than(acc_distributions['Random Forest'],acc_distributions['Logistic Regression'])

    # now testing how many n makes sense
    rf_test=IV(X_balanced, y_balanced, RandomForestClassifier())

    rf_test.get_development("acc", plot=True)






def part4():
    print("\n=== PART 4: Development curve (Balanced Accuracy vs. Training Set Size) ===")
    max_trainset_size = 100

    # (Re)initialize the SVM IV instance.
    iv_dev = IV(X_balanced, y_balanced, SVC(gamma='scale'))
    iv_dev.compute_posterior(burn_in=1000, thin=10, step_size=0.1, num_samples=1000)
    trainset_sizes = np.arange(1, max_trainset_size + 1)
    '''
    
    mean_acc_list = []
    lower_bound_list = []
    upper_bound_list = []
    
    
    # Loop over training set sizes.
    for n in trainset_sizes:
        # Run IV for training set size n.
        # (Here we assume run_iv accepts a parameter `trainset_size` to simulate performance with n samples.)
        iv_dev.run_iv(trainset_size=n)
        acc_dist_n = iv_dev.get_acc_dist(plot=False)
        mean_acc_list.append(np.mean(acc_dist_n))
        # Compute a 95% interval (using the 2.5th and 97.5th percentiles).
        lower_bound_list.append(np.percentile(acc_dist_n, 2.5))
        upper_bound_list.append(np.percentile(acc_dist_n, 97.5))
    '''
    mean_acc_list, lower_bound_list, upper_bound_list = iv_dev.get_development(key='acc', n=101, plot=False, confidence_range=0.5)

    # Plot the development curve.
    plt.figure(figsize=(10, 6))
    plt.plot(trainset_sizes, mean_acc_list, label='Mean Accuracy', color='blue')
    plt.fill_between(trainset_sizes, lower_bound_list, upper_bound_list, color='gray', alpha=0.3, label='50% Credible Interval')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Development Curve: Accuracy vs. Training Set Size (SVM)")
    plt.legend()
    plt.savefig('plots/figure4.png')
    plt.show()

import warnings
warnings.filterwarnings('ignore')
print('setup done')
part1()
part2()
part3()
part4()