import os
from bs4 import BeautifulSoup
import re

def clean_html_text(html_text: str):
    """Function to clean html files, this is then used directly in the write_clean_html_text_files function.

    Args:
        html_text (string): content of the html file

    Returns:
        text (string): returns the cleaned text 
    """

    soup = BeautifulSoup(html_text, 'html.parser')
    
    # Remove elements with specific styles for subscript and superscript
    for tag in soup.find_all(style=True):
        style = tag['style'].replace(' ', '').lower()
        if 'vertical-align:baseline' in style or 'vertical-align:sub' in style or 'vertical-align:super' in style:
            tag.decompose()
    
    # Extract text while preserving spaces between elements
    text = ' '.join(soup.stripped_strings)
    
    # Replace tabs and new lines with a space
    text = re.sub(r'[\t\n]', ' ', text)

    # Replace full stops at the end of the sentence with a space. Letters in abbreviations are concatenated
    pattern3 = re.compile("\.$")
    text = pattern3.sub(' ',text)

    # Remove apostrophes
    text = text.replace("'", "")

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def write_clean_html_text_files(input_folder: str, dest_folder: str):
    """Function to take html files, clean them using the clean_html_text function and turn them into .txt files. The .txt
    files are then read in and split according to a phrase, this is so we can get rid of the table of contents.
    Files can be read easily and saved in destination folder.

    Args:
        input_folder (string): folder to search for html files (e.g. 'C:\Documents\AAPL_html_files')
        dest_folder (string): desired name of the folder containing the output .txt files (e.g. AAPL_cleaned_txt_files')
    
    Returns: 
        None
    """
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
        
        
        with open(f"{dest_folder}/{txt_filename}", 'r') as f:
            content = f.read()
            # Find all positions of "PART I Item 1 Business"
            matches = [match.start() for match in re.finditer(r'\bPART I Item 1 Business\b', content, re.IGNORECASE)]
            
            if len(matches) > 1:
                # If there is more than one occurrence, use the second occurrence
                split_pos = matches[1]
                cleaned_text = content[split_pos:].strip()
            elif matches:
                # If there is only one occurrence, use the first one
                split_pos = matches[0]
                cleaned_text = content[split_pos:].strip()
            else:
                # If the string is not found, return the original content
                cleaned_text = content.strip()
            
            with open(f"{dest_folder}/{txt_filename}", 'w', encoding="utf-8") as output_file:
                output_file.write(cleaned_text)