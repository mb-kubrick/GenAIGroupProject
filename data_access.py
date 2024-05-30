import os
import pandas as pd
import requests
from datetime import datetime, timedelta

def download_files_10k(ticker: str, dest_folder: str)->None:
    """Function to download the last 3 years of 10-K filings for S&P100
        companies to the designated folder.

    Args:
        ticker (str): company ticker for S&P100 company (e.g. 'AAPL')
        dest_folder (str): desired destination folder of the file (e.g. ".\AAPL_html_files")
    
    Returns:
        None
    """
   
    # Load company data - ticker and CIK number (with and without zeros)
    SP100data = pd.read_csv("data/companyData.csv", index_col=False)
    SP100data = SP100data.set_index("ticker")
    SP100data["CIK Zeros"] = SP100data["CIK Zeros"].astype(str).str.zfill(10)
    
    ticker = ticker
    cik = SP100data.loc[ticker]["CIK"]
    cik_zero = SP100data.loc[ticker]["CIK Zeros"]
    
    # Create request header
    headers = {"User-Agent": "your-email@example.com"}
    
    # Get all file submission metadata
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
    
    # Filter for the last 3 years
    three_years_ago = datetime.now() - timedelta(days=3*365)
    forms_10k["reportDate"] = pd.to_datetime(forms_10k["reportDate"])
    recent_forms_10k = forms_10k[forms_10k["reportDate"] >= three_years_ago]

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    
    # Loop over the recent 10-K reports to download them
    for i in range(len(recent_forms_10k)):
        doc_link = recent_forms_10k["primaryDocument"].iloc[i]
        accessionNumber = recent_forms_10k["accessionNumber"].iloc[i]
        accessionNumber = accessionNumber.replace("-", "")
        form_type = recent_forms_10k["form"].iloc[i]
        report_date = recent_forms_10k["reportDate"].iloc[i].strftime('%Y-%m-%d')
        
        file_path = os.path.join(dest_folder, f"{ticker}-{form_type}-{report_date}.html")
        file = requests.get(
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessionNumber}/{doc_link}",
            headers=headers,
        )
        
        with open(file_path, "wb") as f:
            f.write(file.content)
