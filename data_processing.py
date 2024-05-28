from bs4 import BeautifulSoup
import re

def clean_html_text(html_text: str):
    """Function to clean html files, this is then used directly in the write_clean_html_text_files function.

    Args:
        html_text (string): content of the html file

    Returns:
        cleaner_text (string): returns the cleaned text 
    """

    # Create soup object
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove tags
    text = soup.get_text()

    # Remove bullet points
    cleaned_text = re.sub(r"\s*-\s+", " ", text.strip())

    # Remove special characters and extra whitespace
    cleaner_text = re.sub(r"[^\w\s]", "", cleaned_text).replace("\n", "")

    return cleaner_text

def write_clean_html_text_files(input_folder: str, dest_folder: str):
    """Function to take html files, clean them using the clean_html_text function and turn them into .txt files.
    Files can be read easily and saved in destination folder.

    Args:
        input_folder (string): folder to search for html files (e.g. 'C:\Documents\AAPL_html_files')
        dest_folder (string): desired destination and name of the folder containing the output .txt files ((e.g. 'C:\Documents\AAPL_cleaned_txt_files'))
    
    Returns: 
        None
    """
    # Import packages
    import os

    # Create destination folder if it does not already exist
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    # Retrieve .html files only from input folder
    html_files = []
    files_list = os.listdir(input_folder)
    for filename in files_list:
        if filename.endswith(".html") or filename.endswith(".htm"):
            html_files.append(filename)

    # Clean all retrieved .html files
    for html_filename in html_files:
        # encoding = 'cp437' specificies which charset should be used to read chars
        # errors ='ignore' will ignore any special chars that can't be decoded by the specified charset
        my_file = open(
            f"{input_folder}/{html_filename}", "r", encoding="utf-8", errors="ignore"
        )
        txt = my_file.read()
        my_file.close()
        cleaned_txt = clean_html_text(txt)

        # Rename .html file as .txt and save in destination folder
        txt_filename = None
        if html_filename.endswith(".html"):
            txt_filename = html_filename.replace(".html", ".txt")
        if html_filename.endswith(".htm"):
            txt_filename = html_filename.replace(".htm", ".txt")

        with open(
            f"{dest_folder}/{txt_filename}", "w", encoding="utf-8"
        ) as output_file:
            output_file.write(cleaned_txt)
