import sqlite3
import pandas as pd
import os

# --- Configuration: Point to the local database file ---
DB_FILENAME = "chembl_35.db"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, '..', 'data', DB_FILENAME)


def find_targets_for_disease(keyword, max_results=3):
    """
    Finds potential protein targets for a given disease keyword by querying
    the local SQLite ChEMBL database.
    """
    print(f"Searching for targets related to '{keyword}' in local database...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at {DB_PATH}. Please ensure it is in the 'data' folder.")

    # CORRECTED QUERY: Searches only the primary name field in target_dictionary,
    # which is more stable across ChEMBL versions.
    query = """
            SELECT DISTINCT td.chembl_id AS target_chembl_id, \
                            td.pref_name
            FROM target_dictionary td
            WHERE td.pref_name LIKE ? LIMIT ?; \
            """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn, params=(f'%{keyword}%', max_results))
        conn.close()
        return df
    except Exception as e:
        print(f"Error querying local database for targets: {e}")
        return pd.DataFrame()


def fetch_compounds_for_target(target_chembl_id, max_compounds=200):
    """
    Fetches active compounds for a specific target ChEMBL ID from the local
    SQLite database. This version uses the correct join path through the 'assays' table.
    """
    print(f"Fetching compounds for target '{target_chembl_id}' from local database...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at {DB_PATH}. Please ensure it is in the 'data' folder.")

    # CORRECTED QUERY: This joins 'activities' to 'assays' and then to 'targets',
    # which is the correct and most robust way to link these tables in the ChEMBL schema.
    query = """
            SELECT md.chembl_id                        AS molecule_chembl_id, \
                   cs.canonical_smiles                 AS smiles, \
                   act.standard_type, \
                   act.standard_value, \
                   act.pchembl_value, \
                   act.standard_units, \
                   (1.0 / (act.standard_value + 1e-6)) AS rank_score
            FROM activities act
                     JOIN molecule_dictionary md ON act.molregno = md.molregno
                     JOIN compound_structures cs ON md.molregno = cs.molregno
                     JOIN assays a ON act.assay_id = a.assay_id
                     JOIN target_dictionary td ON a.tid = td.tid
            WHERE td.chembl_id = ?
              AND act.standard_type = 'IC50'
              AND act.standard_value IS NOT NULL
            ORDER BY act.standard_value ASC LIMIT ?; \
            """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn, params=(target_chembl_id, max_compounds))
        conn.close()

        if df.empty:
            print(f"No compounds with IC50 values found for target {target_chembl_id}.")
            return pd.DataFrame()

        return df
    except Exception as e:
        print(f"Error querying local database for compounds: {e}")
        return pd.DataFrame()