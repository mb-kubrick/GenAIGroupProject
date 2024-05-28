import requests
import pandas as pd
import os

def download_files_10k(ticker: str, dest_folder: str):
    """Function to download all 10-K filings for S&P100
        companies to designated folder.
    Arguments:
    ---------------------------------------------------
    - ticker: type = string, company ticker for S&P100 company
    (e.g. 'AAPL')
    - dest_folder: type = string, desired destination folder of the file
      (e.g. 'C:\Documents\mormont-edgar')
    """
    # Find company data - ticker and CIK number (with and without zeros)
    SP100data = pd.read_csv("companyData.csv", index_col=False)
    SP100data = SP100data.set_index("ticker")
    SP100data["CIK Zeros"] = SP100data["CIK Zeros"].astype(str).str.zfill(10)

    ticker = ticker
    cik = SP100data.loc[ticker]["CIK"]
    cik_zero = SP100data.loc[ticker]["CIK Zeros"]

    # Create request header
    headers = {"User-Agent": "fahimaahmed@kubrickgroup.com"}

    # Cet all file submission metadata
    company_files = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik_zero}.json", headers=headers
    )

    # Find all recent filings
    allForms = pd.DataFrame.from_dict(company_files.json()["filings"]["recent"])

    # Select all 10-K reports
    mask = allForms["form"] == "10-K"
    forms_10k = allForms[mask]
    forms_10k = forms_10k[
        ["accessionNumber", "reportDate", "form", "primaryDocument"]
    ].reset_index(drop=True)

    # Create destination folder if it does not already exist
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    # Loop over all 10-K reports to download them
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