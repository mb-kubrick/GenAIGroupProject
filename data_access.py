import requests
import pandas as pd
import os

def download_files_10k(ticker: str, dest_folder: str) -> None:
    """Function to download all 10-K filings for S&P100 companies to a designated folder.

    Gets all file submission metadata. Finds all recent filings. Select all 10-K reports. Loop over all 10-K reports to
    download them. Writes to file.

    Args:
        ticker (str): Company ticker for the given S&P100 company.
        dest_folder (str): desired destination and name of folder (e.g. 'C:\Documents\AAPL_html_files').
    
    Returns:
        None
    """

    # Reads the CSV containing the list of all the tickers, their CIK (Central Index Key, unique identifier) and 
    # CIK Zeros (CIK with padding zeros at the beginning if needed, used to construct URLs for API requests, SEC EDGAR system requires CIK numbers to be 10 digits)
    SP100data = pd.read_csv("data\companyData.csv", index_col=False)
    SP100data = SP100data.set_index("ticker")
    SP100data["CIK Zeros"] = SP100data["CIK Zeros"].astype(str).str.zfill(10)

    # Identifying ticker name, its CIK and CIK Zeros 
    ticker = ticker
    cik = SP100data.loc[ticker]["CIK"]
    cik_zero = SP100data.loc[ticker]["CIK Zeros"]

    # Need to specify who is requesting the data, usually an email address so SEC can get into contact if there are issues or changes
    headers = {"User-Agent": "fahimaahmed@kubrickgroup.com"}
    
    # Get the json files for the company using requests, these files contain information about the reports
    company_files = requests.get(f"https://data.sec.gov/submissions/CIK{cik_zero}.json", headers=headers)

    # Dataframe of all the recent filings, mask to get all the 10 K files
    allForms = pd.DataFrame.from_dict(company_files.json()["filings"]["recent"])
    mask = allForms["form"] == "10-K"
    forms_10k = allForms[mask]
    forms_10k = forms_10k[["accessionNumber", "reportDate", "form", "primaryDocument"]].reset_index(drop=True)

    # Create destination folder if it does not already exist
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    
    # Loop through all the 10 K files, get the required information to construct the URL, extracted using requests and saved at html files in desired folder
    for i in range(0, len(list(forms_10k["primaryDocument"]))):
        doc_link = forms_10k["primaryDocument"][i]
        accessionNumber = forms_10k["accessionNumber"][i]
        accessionNumber = accessionNumber.replace("-", "")
        form_type = forms_10k["form"][i]
        report_date = forms_10k["reportDate"][i]

        file_path = dest_folder + f"\{ticker}-{form_type}-{report_date}.html"
        file = requests.get(
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessionNumber}/{doc_link}",
            headers=headers,
        )

        with open(file_path, "wb") as f:
            f.write(file.content)
            f.close()
