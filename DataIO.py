
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import pickle
sns.set(color_codes=True)


def load_file(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x

def load_data(train_rate=85,val_rate=5):
    X = load_file('data/img')
    y = load_file('data/label')
    if (100-train_rate-val_rate<=0):
        return None
    num_train = int(X.shape[0] / 100 * train_rate)
    num_val = int(X.shape[0] / 100 * val_rate)
    train_images = []                                                   # reshape train images so that the training set
    for i in range(num_train+num_val):                                   # is of shape (x, 1, 28, 28)
        train_images.append(np.expand_dims(X[i], axis=0))
    train_images = np.array(train_images)

    test_images = []                                                    # reshape test images so that the test set
    for i in range(num_train+num_val,X.shape[0]):                                    # is of shape (x, 1, 28, 28)
        test_images.append(np.expand_dims(X[i], axis=0))
    test_images = np.array(test_images)

    train_labels = []
    for i in range(num_train+num_val):                               
        train_labels.append(y[i])
    train_labels = np.array(train_labels)

    test_labels = []
    for i in range(num_train+num_val,y.shape[0]):                               
        test_labels.append(y[i])
    test_labels = np.array(test_labels)

    indices = np.random.permutation(train_images.shape[0])              # permute and split training data in
    training_idx, validation_idx = indices[:num_train], indices[num_train:]     # training and validation sets
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('training_accuracy.png')
    plt.show()


def plot_learning_curve(loss_history, val_loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Train Loss')
    plt.plot(val_loss_history, 'r', linewidth=3.0, label='Val Loss')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('learning_curve.png')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()
