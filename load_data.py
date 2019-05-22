import numpy as np

def load_subsets(x, y, num_per_class=15, num_classes=100):
    """
    targets should NOT be one-hot encoded
    """
    x_subset = [[]] * num_classes
    y_subset = [[]] * num_classes
    total = 0
    size = len(x)
    assert ((size > num_per_class * num_classes) and (size / num_per_class >= num_classes * num_per_class)),"Not enough data to create subset."

    for i in range(size):
        if not total < num_per_class * num_classes:
            break
        for j in range(num_classes):
            if len(y_subset[j]) < num_per_class:
                x_subset[j].append(x[i])
                y_subset[j].append(y[i])
                total += 1
                break

    return np.vstack(x_subset), np.vstack(y_subset)
