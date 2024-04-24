from sklearn.metrics import cohen_kappa_score

def calculate_cohens_kappa(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

def main():
    # Example data
    # y_true = [0, 1, 0, 0, 1]
    # y_pred = [0, 1, 1, 0, 1]

    # Calculate Cohen's kappa
    kappa = calculate_cohens_kappa(y_true, y_pred)
    print("Cohen's kappa:", kappa)

if __name__ == "__main__":
    main()
