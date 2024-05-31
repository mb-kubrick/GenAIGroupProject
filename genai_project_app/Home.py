import streamlit as st

st.title("GenAi final project")
st.write("<span style='font-style: italic;'>Fahima Ahmed, Michael Berney, Mak Dedic, Kiran Hosein, Danielle Hurford, Visahan Sritharan (Pod 1)</span>", unsafe_allow_html=True)

st.header("Project Brief")

st.markdown(
    f'<div style="border: 2px solid #ADD8E6; padding: 10px; background-color: #ADD8E6; text-align: center;">'
    f'<p style="color: black; margin: 0;">Stimulate an Investment Management role by leveraging an AI Assistant that is powered by an LLM</p>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(""" """)
st.markdown(
    """
    Ai Assistant must:

    - Query open-sourced financial data (10-K Annual reports)
    - Perform web searched to gather latest company information
    - Rebalancing portfolio weights and updating investment information based on the insights gained from the data analysis

    """
)

st.header("Project Workflow")

st.write("Created and allocated tickets on Jira")

st.subheader("Data Pre-Processing")

data_preprocessing = st.button("🧹")

if data_preprocessing:  
    st.markdown(
        """
        🧹 Download the 10-K files for chosen tickers (companies) using the SEC EDGAR API for the last 3 years
        🧹 Clean .html files by removing html tags and special characters
        """
    )

st.subheader("Agent")

agent = st.button("🔧", type="primary")

if agent:  
    st.markdown(
        """
        🔧 SQL Database
            - create synthetic data by allocating each ticker with random percentages
        🔧 Vector Database
        🔧 Web Search
        """
    )
