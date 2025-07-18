import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import streamlit as st
from contextlib import contextmanager
import numpy as np

@contextmanager
def sqlite_connection(db_path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def create_cell_count_db(csv_path: str, db_path: str):
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create subjects table
    subjects_df = df[['subject', 'project', 'condition', 'age', 'sex', 'treatment', 'response']].drop_duplicates()
    subjects_df.columns = ['subject_id', 'project', 'condition', 'age', 'sex', 'treatment', 'response']
    subjects_df.to_sql('subjects', conn, if_exists='replace', index=False)

    # Create samples table
    samples_df = df[['sample', 'subject', 'sample_type', 'time_from_treatment_start',
                     'b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']]
    samples_df.columns = ['sample_id', 'subject_id', 'sample_type', 'time_from_treatment_start',
                          'b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
    samples_df.to_sql('samples', conn, if_exists='replace', index=False)

    # Get counts for db verification
    cursor.execute("SELECT COUNT(*) FROM subjects;")
    subject_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM samples;")
    sample_count = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    return subject_count, sample_count

def calculate_population_percentages(db_path: str):
    # get data from the db
    with sqlite_connection(db_path) as conn:
        query = """
        SELECT sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte
        FROM samples;
        """
        df_samples = pd.read_sql_query(query, conn)

    # Melt the dataframe to long format
    melted = df_samples.melt(
        id_vars=['sample_id'],
        value_vars=['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte'],
        var_name='population',
        value_name='count'
    )

    # Compute total count per sample
    total_counts = melted.groupby('sample_id')['count'].sum().reset_index()
    total_counts.columns = ['sample_id', 'total_count']

    # Merge and compute percentage
    merged = melted.merge(total_counts, on='sample_id')
    merged['percentage'] = (merged['count'] / merged['total_count']) * 100
    merged['percentage'] = merged['percentage'].round(3)

    final_df = merged[['sample_id', 'total_count', 'population', 'count', 'percentage']]

    return final_df

def analyze_immune_response(population_df: pd.DataFrame, db_path: str):
    # get data from the db
    with sqlite_connection(db_path) as conn:
        combined_df = pd.read_sql_query(
        "SELECT * FROM samples JOIN subjects ON samples.subject_id = subjects.subject_id;",
        conn
        )

    # Merge all data
    merged_df = pd.merge(population_df, combined_df, on='sample_id')

    # Filter for PBMC samples from melanoma patients treated with miraclib
    filtered_df = merged_df[
        (merged_df['sample_type'] == 'PBMC') &
        (merged_df['condition'] == 'melanoma') &
        (merged_df['treatment'] == 'miraclib')
    ]

    # Create boxplot
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='population', y='percentage', hue='response', data=filtered_df, ax=ax)
    ax.set_title('Immune Cell Population Frequencies: Responders vs Non-Responders')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    # Perform Mann-Whitney U tests
    results = []
    populations = filtered_df['population'].unique()
    n_tests = len(populations)

    for pop in populations:
        group = filtered_df[filtered_df['population'] == pop]
        responders = group[group['response'] == 'yes']['percentage']
        non_responders = group[group['response'] == 'no']['percentage']
        
        try:
            assert len(group) == (len(responders) + len(non_responders))
        except AssertionError:
            print("Warning, not all responses are yes or no.")

        # Data is formatted corrected, two distict populations
        if len(responders) > 0 and len(non_responders) > 0:
            stat, p = mannwhitneyu(responders, non_responders, alternative='two-sided')
            p_adj = min(p * n_tests, 1.0)

            results.append({
                'population': pop,
                'median_responders': responders.median(),
                'median_non_responders': non_responders.median(),
                'p_value': p,
                'p_value_adj': p_adj,
                'significant': p_adj < 0.05
            })
        
        # If there are no responders or no non-responders, fill in table where possible
        else:
            print(f"Warning: Not enough data for population '{pop}'. Responders: {len(responders)}, Non-responders: {len(non_responders)}")
            results.append({
                'population': pop,
                'median_responders': responders.median() if len(responders) > 0 else np.nan,
                'median_non_responders': non_responders.median() if len(non_responders) > 0 else np.nan,
                'p_value': np.nan,
                'p_value_adj': np.nan,
                'significant': False
            })

    results_df = pd.DataFrame(results).sort_values(by='p_value')

    return results_df, fig

def summarize_melanoma_baseline_samples(db_path: str = "cell_count.db"):
    # get data from the db
    # Identify all melanoma PBMC samples at baseline
    with sqlite_connection(db_path) as conn:
        filtered_query = """
        SELECT 
            s.sample_id,
            subj.subject_id,
            subj.project,
            subj.response,
            subj.sex,
            subj.condition,
            s.sample_type,
            s.time_from_treatment_start,
            subj.treatment
        FROM samples s
        JOIN subjects subj ON s.subject_id = subj.subject_id
        WHERE subj.condition = 'melanoma'
        AND s.sample_type = 'PBMC'
        AND s.time_from_treatment_start = 0
        AND subj.treatment = 'miraclib'
        """
        filtered_samples = pd.read_sql_query(filtered_query, conn)


    # Among these samples, extend the query to determine:
    # How many samples from each project
    # How many subjects were responders/non-responders 
    # How many subjects were males/females
    with sqlite_connection(db_path) as conn:
        summary_query = """
        WITH filtered AS (
            SELECT 
                s.sample_id,
                subj.subject_id,
                subj.project,
                subj.response,
                subj.sex
            FROM samples s
            JOIN subjects subj ON s.subject_id = subj.subject_id
            WHERE subj.condition = 'melanoma'
            AND s.sample_type = 'PBMC'
            AND s.time_from_treatment_start = 0
            AND subj.treatment = 'miraclib'
        )
        SELECT 'samples_per_project' AS category, project AS label, COUNT(sample_id) AS value
        FROM filtered
        GROUP BY project

        UNION ALL

        SELECT 'subjects_by_response' AS category, response AS label, COUNT(DISTINCT subject_id) AS value
        FROM filtered
        GROUP BY response

        UNION ALL

        SELECT 'subjects_by_sex' AS category, sex AS label, COUNT(DISTINCT subject_id) AS value
        FROM filtered
        GROUP BY sex;
        """
        summary_df = pd.read_sql_query(summary_query, conn)

    return filtered_samples, summary_df

def launch_dashboard(
    summary_table_part2: pd.DataFrame,
    summary_table_part3: pd.DataFrame,
    summary_table_part4: pd.DataFrame,
    filtered_table_part4: pd.DataFrame,
    fig: plt.Figure
):
    st.title("Immune Cell Analysis Dashboard")

    # Define tabs
    tab1, tab2, tab3 = st.tabs([
        "Cell Type Frequencies",
        "Statistical Analysis",
        "Subset Analysis & Filters"
    ])

    # tab 1
    with tab1:
        st.header("Cell Type Frequencies by Sample")

        sample_selected = st.selectbox("Select a Sample ID", summary_table_part2['sample_id'].unique())
        sample_data = summary_table_part2[summary_table_part2['sample_id'] == sample_selected]

        st.write("### Frequency Table for Selected Sample")
        st.dataframe(sample_data)

    # tab 2
    with tab2:
        st.header("Statistical Analysis of Responders vs Non-Responders")

        st.write("### Boxplot Comparison")
        st.pyplot(fig)

        # Convert boolean to readable text
        # Makes table look better to viewers
        stats_df = summary_table_part3.copy()
        stats_df['Significant'] = stats_df['significant'].map({True: "Yes", False: "No"})
        stats_df_display = stats_df.drop(columns=["significant"])

        st.write("### Statistical Summary Table")
        st.dataframe(stats_df_display)

        # Statistical Analysis Explanation
        st.markdown("""
        Note: Statistical comparisons between responders and non-responders were conducted using the non-parametric Mann-Whitney U test, which does not assume normal distributions.

        Because multiple immune cell types were tested simultaneously, a Bonferroni correction was applied to control for the increased risk of false positives due to multiple comparisons. Specifically, each p-value was adjusted using the formula:  
        `p_adj = min(p × N, 1.0)`  
        where `N` is the number of tests performed.

        Only comparisons with adjusted p-values < 0.05 are considered statistically significant in this analysis.
        """)

        significant_only = stats_df_display[stats_df_display["Significant"] == "Yes"]
        if len(significant_only) > 0:
            st.write("### Summary of Significant Immune Cell Types")
            st.dataframe(significant_only)
        else:
            st.write("### Summary of Significant Immune Cell Types")
            st.write("No immune cell types showed statistically significant differences (adjusted p < 0.05).")

    # tab 3
    with tab3:
        st.header("Subset Analysis: Overview & Filtering")

        st.subheader("Overview Counts")
        for cat in summary_table_part4['category'].unique():
            sub_df = summary_table_part4[summary_table_part4['category'] == cat]
            st.subheader(cat.replace("_", " ").title())
            st.bar_chart(data=sub_df.set_index('label'), y='value')

        st.subheader("Baseline PBMC Melanoma Samples (miraclib)")
        st.write("### Full Filtered Data Table")
        st.dataframe(filtered_table_part4)

        with st.expander("Filter Options"):
            gender = st.multiselect("Sex", filtered_table_part4['sex'].unique(), default=filtered_table_part4['sex'].unique())
            response = st.multiselect("Response", filtered_table_part4['response'].unique(), default=filtered_table_part4['response'].unique())
            project = st.multiselect("Project", filtered_table_part4['project'].unique(), default=filtered_table_part4['project'].unique())

            filtered = filtered_table_part4[
                filtered_table_part4['sex'].isin(gender) &
                filtered_table_part4['response'].isin(response) &
                filtered_table_part4['project'].isin(project)
            ]

            st.write(f"Filtered rows: {len(filtered)}")
            st.dataframe(filtered)

    st.markdown("---")

def main():
    # part 1
    subjects, samples = create_cell_count_db("cell-count.csv", "cell_count.db")
    print("Subjects:", subjects)
    print("Samples:", samples)

    # part 2
    summary_table_part2 = calculate_population_percentages("cell_count.db")
    print(summary_table_part2.head(10))

    # part 3
    summary_table_part3, fig = analyze_immune_response(summary_table_part2, "cell_count.db")
    print(summary_table_part3)
    plt.show()

    # part 4
    filtered_table_part4, summary_table_part4 = summarize_melanoma_baseline_samples("cell_count.db")
    print(filtered_table_part4.head())
    print(summary_table_part4)

    # Launch dashboard
    launch_dashboard(
        summary_table_part2,
        summary_table_part3,
        summary_table_part4,
        filtered_table_part4,
        fig
    )

if __name__ == "__main__":
    main()
