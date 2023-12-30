import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 100 trials of 10-fold cross-validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump accuracy
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    # Number of trials and folds
    num_trials = 100
    num_folds = 10

    # Initialize arrays to store accuracy values
    decision_tree_accuracies = np.zeros(num_trials * num_folds)
    decision_stump_accuracies = np.zeros(num_trials * num_folds)
    dt3_accuracies = np.zeros(num_trials * num_folds)

    for trial in range(num_trials):
        # shuffle the data at the start of each trial
        idx = np.arange(n)
        np.random.seed(trial)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Split the data into folds
        fold_size = n // num_folds
        folds_X = [X[i * fold_size:(i + 1) * fold_size, :] for i in range(num_folds)]
        folds_y = [y[i * fold_size:(i + 1) * fold_size, :] for i in range(num_folds)]

        for fold in range(num_folds):
            # Select the current fold as the test set, and the rest as the training set
            X_test = folds_X[fold]
            y_test = folds_y[fold]
            X_train = np.vstack(folds_X[:fold] + folds_X[fold + 1:])
            y_train = np.vstack(folds_y[:fold] + folds_y[fold + 1:])

            # Train the decision tree
            clf_tree = DecisionTreeClassifier()
            clf_tree.fit(X_train, y_train)

            # Train the decision stump (1-level decision tree)
            clf_stump = DecisionTreeClassifier(max_depth=1)
            clf_stump.fit(X_train, y_train)

            # Train the 3-level decision tree
            clf_dt3 = DecisionTreeClassifier(max_depth=3)
            clf_dt3.fit(X_train, y_train)

            # Evaluate the models on the test set
            y_pred_tree = clf_tree.predict(X_test)
            y_pred_stump = clf_stump.predict(X_test)
            y_pred_dt3 = clf_dt3.predict(X_test)

            # Store the accuracy values
            decision_tree_accuracies[trial * num_folds + fold] = accuracy_score(y_test, y_pred_tree)
            decision_stump_accuracies[trial * num_folds + fold] = accuracy_score(y_test, y_pred_stump)
            dt3_accuracies[trial * num_folds + fold] = accuracy_score(y_test, y_pred_dt3)

    # Calculate mean and standard deviation for each classifier
    mean_decision_tree_accuracy = np.mean(decision_tree_accuracies)
    std_decision_tree_accuracy = np.std(decision_tree_accuracies)
    mean_decision_stump_accuracy = np.mean(decision_stump_accuracies)
    std_decision_stump_accuracy = np.std(decision_stump_accuracies)
    mean_dt3_accuracy = np.mean(dt3_accuracies)
    std_dt3_accuracy = np.std(dt3_accuracies)

    # Prepare the result matrix
    stats = np.zeros((3, 2))
    stats[0, 0] = mean_decision_tree_accuracy
    stats[0, 1] = std_decision_tree_accuracy
    stats[1, 0] = mean_decision_stump_accuracy
    stats[1, 1] = std_decision_stump_accuracy
    stats[2, 0] = mean_dt3_accuracy
    stats[2, 1] = std_dt3_accuracy

    # Plot learning curves
    plot_learning_curves(decision_tree_accuracies, decision_stump_accuracies, dt3_accuracies)

    return stats

def plot_learning_curves(decision_tree_accuracies, decision_stump_accuracies, dt3_accuracies):
    '''
    Plot the learning curves based on the provided accuracy values.

    Parameters:
      decision_tree_accuracies (numpy array): Array containing decision tree accuracy values
      decision_stump_accuracies (numpy array): Array containing decision stump accuracy values
      dt3_accuracies (numpy array): Array containing 3-level decision tree accuracy values
    '''
    mean_curve_tree = np.mean(decision_tree_accuracies.reshape(-1, 10), axis=0)
    std_curve_tree = np.std(decision_tree_accuracies.reshape(-1, 10), axis=0)

    mean_curve_stump = np.mean(decision_stump_accuracies.reshape(-1, 10), axis=0)
    std_curve_stump = np.std(decision_stump_accuracies.reshape(-1, 10), axis=0)

    mean_curve_dt3 = np.mean(dt3_accuracies.reshape(-1, 10), axis=0)
    std_curve_dt3 = np.std(dt3_accuracies.reshape(-1, 10), axis=0)

    x_points = np.arange(10) * 10 + 10  # 10%, 20%, ..., 100%

    plt.errorbar(x_points, mean_curve_tree, yerr=std_curve_tree, label='Decision Tree', marker='o')
    plt.errorbar(x_points, mean_curve_stump, yerr=std_curve_stump, label='Decision Stump', marker='o')
    plt.errorbar(x_points, mean_curve_dt3, yerr=std_curve_dt3, label='3-level Decision Tree', marker='o')

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = {:.4f} (± {:.4f})".format(stats[0, 0], stats[0, 1]))
    print("Decision Stump Accuracy = {:.4f} (± {:.4f})".format(stats[1, 0], stats[1, 1]))
    print("3-level Decision Tree Accuracy = {:.4f} (± {:.4f})".format(stats[2, 0], stats[2, 1]))
# ...to HERE.
