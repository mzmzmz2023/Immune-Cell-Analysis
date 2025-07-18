Instructions:
1. Ensure you have Python 3.8+ and install required packages:
pip install pandas numpy streamlit seaborn matplotlib scipy

2. Ensure the CSV file cell-count.csv is in the same directory as immune_dashboard.py

3. run "streamlit run immune_cell_analysis.py" in your terminal

Schema Design:
The relational database schema consists of two primary tables: subjects and samples. The subjects table captures unique individuals and their metadata such as project affiliation, medical condition, demographics, treatment, and response. Each subject has a unique subject_id. The samples table stores observations linked to these subjects, including timepoints and measured immune cell counts (B cells, T cells, etc.). Each entry in samples has a unique sample_id and a foreign key subject_id to maintain referential integrity with the subjects table. This schema separates subject-level metadata from sample-level data, reducing redundancy and improving consistency. As the dataset scales to hundreds of projects and thousands of samples, it remains simple to analzye data in smaller subsets by filtering by project. 

Code structure:
The code consists of 6 functions: 4 helper functions that each handle one part of the project, another helper function to build the dashboard, and the main function to call all the helper functions.

The code is in discrete, modular units making for easy testing and maintaining between different parts.

Dashboard link:

