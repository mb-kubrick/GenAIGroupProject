from bs4 import BeautifulSoup
import re

def clean_html_text(html_text):
    """Function to clean html files
    Arguments:
    ---------------------------------------------------
    - html_text: type = string, location of html file
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

def write_clean_html_text_files(input_folder, dest_folder):
    # import packages
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
