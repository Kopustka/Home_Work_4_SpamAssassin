import os
import glob

def get_datas(spam_path, ham_path):
    """
    Retrieves the list of email files from the given directories.

    Parameters:
        spam_path (str): The directory path containing spam email files.
        ham_path (str): The directory path containing ham (non-spam) email files.

    Returns:
        tuple: (spam_files, ham_files)
        - spam_files (list): List of file paths for spam emails.
        - ham_files (list): List of file paths for ham emails.
    """

    # Get all spam email file paths from the given directory
    spam_files = glob.glob(os.path.join(spam_path, "*"))

    # Get all ham (non-spam) email file paths from the given directory
    ham_files = glob.glob(os.path.join(ham_path, "*"))

    return spam_files, ham_files
