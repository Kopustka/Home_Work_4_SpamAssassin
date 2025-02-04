import random

def balancing(spam_files, ham_files):
    """
    Balances the dataset by ensuring the number of spam and ham files is approximately equal.
    If there are too many ham files, some are randomly removed.
    If there are too many spam files, some are also randomly removed.

    Parameters:
        spam_files (list): List of paths to spam email files.
        ham_files (list): List of paths to ham (non-spam) email files.

    Returns:
        tuple: Balanced lists of spam and ham files.
    """

    # Determine the target number of ham files (spam count + random adjustment of 30-40 files)
    target_ham_count = len(spam_files) + random.randint(30, 40)

    # Determine the target number of spam files (ham count + random adjustment of 30-40 files)
    target_spam_count = len(ham_files) + random.randint(30, 40)

    # If there are too many ham files, randomly select only the required amount
    if len(ham_files) > target_ham_count:
        ham_files = random.sample(ham_files, target_ham_count)

    # If there are too many spam files, randomly select only the required amount
    if len(spam_files) > target_spam_count:
        spam_files = random.sample(spam_files, target_spam_count)

    return spam_files, ham_files

