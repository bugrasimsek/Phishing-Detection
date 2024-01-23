import os

sourceFolders = {
    'benign': 'PhishIntention/benign_25k',
    'phishing': 'PhishIntention/phish_sample_30k',
    'misleading': 'PhishIntention/misleading'
}

targetFolders = {
    'legitimate': 'Legitimate',
    'phishing': 'Phishing'
}


def move_files(source_folder, target_folder, file_extension):
    # Create target directory if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Loop through each folder in the source directory
    for folder in os.listdir(source_folder):
        for file in os.listdir(os.path.join(source_folder, folder)):
            if file == file_extension:
                # Rename the file and move it to the target directory
                os.rename(os.path.join(source_folder, folder, file),
                          os.path.join(target_folder, folder + "_" + file))


# Get source folders from dictionary
benignSource = sourceFolders['benign']
phishingSource = sourceFolders['phishing']
misleadingSource = sourceFolders['misleading']

# Get target folders from dictionary
legitTarget = targetFolders['legitimate']
phishTarget = targetFolders['phishing']

# Call the move files function to move files from source to target folders
move_files(benignSource, legitTarget, 'html.txt')
move_files(misleadingSource, legitTarget, 'html.txt')
move_files(phishingSource, phishTarget, 'html.txt')
