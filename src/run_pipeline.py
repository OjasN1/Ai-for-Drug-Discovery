import argparse
import os
import pandas as pd
from src.chembl_api import find_targets_for_disease, fetch_compounds_for_target
from src.datasets import graphs_from_csv, make_loaders
from src.train_local import LocalTrainer
from src.fragment_mix import propose_new_compound
from src.utils import calculate_drug_likeness
import traceback
import torch


def main():
    try:
        # --- Detect CPU cores for speed ---
        try:
            num_workers = os.cpu_count()
            print(f"--- Detected {num_workers} CPU cores. Will use for parallel data loading. ---")
        except:
            num_workers = 0
            print("--- Could not detect CPU cores. Using single-core data loading. ---")

        ap = argparse.ArgumentParser()
        ap.add_argument("--disease_or_target", type=str, required=True,
                        help="Disease keyword (e.g., 'breast cancer') or ChEMBL target ID (e.g., CHEMBL203)")
        ap.add_argument("--is_target_id", action="store_true", help="Interpret disease_or_target as a target ChEMBL ID")
        ap.add_argument("--max_known", type=int, default=200)
        ap.add_argument("--epochs", type=int, default=10)
        ap.add_argument("--lam", type=float, default=1.0, help="tradeoff for (yield - lam * toxicity)")
        args = ap.parse_args()

        print(f"Starting pipeline with args: {args}")

        # --- Phase 1: Data Acquisition ---
        if args.is_target_id:
            targets_df = pd.DataFrame([{"target_chembl_id": args.disease_or_target, "pref_name": "user_provided"}])
        else:
            targets_df = find_targets_for_disease(args.disease_or_target, max_results=3)

        print(f"Found targets:\n{targets_df}")
        if targets_df.empty:
            print("No targets found for keyword. Try --is_target_id with a known target like CHEMBL203")
            return

        chosen_target = targets_df.iloc[0]["target_chembl_id"]
        print(f"Using target: {chosen_target} ({targets_df.iloc[0].get('pref_name', '')})")

        known_df = fetch_compounds_for_target(chosen_target, max_compounds=args.max_known)
        if known_df.empty:
            print("No compounds found for target")
            return

        print(f"Fetched {len(known_df)} compounds")
        top_row = known_df.iloc[0]
        print("Best-known compound (by IC50):")
        print({
            "molecule_chembl_id": top_row.molecule_chembl_id,
            "smiles": top_row.smiles,
            "pchembl_value": top_row.pchembl_value,
            "standard_type": top_row.standard_type,
            "standard_value": top_row.standard_value,
            "units": top_row.standard_units
        })

        # --- Phase 2: Data Preparation ---
        tmp_csv = "data/known_compounds.csv"

        # --- MODIFIED: Add rank_score to demo df for filtering ---
        demo = known_df[["smiles", "rank_score"]].copy()

        rank = (demo["rank_score"] - demo["rank_score"].min()) / max(
            demo["rank_score"].max() - demo["rank_score"].min(), 1e-6)
        demo["yield"] = 0.5 + 0.5 * rank
        demo["tox"] = 0.5 * (1.0 - rank)

        os.makedirs("data", exist_ok=True)
        demo.to_csv(tmp_csv, index=False)
        print(f"Saved synthetic labels to {tmp_csv}")

        # --- Phase 3: Model Training ---
        data_list = graphs_from_csv(tmp_csv, smiles_col="smiles", target_col="yield", default_target=0.5)

        tox_map = {row['smiles']: float(row["tox"]) for _, row in demo.iterrows()}
        valid_data_list = []
        for d in data_list:
            smi = d.smiles
            if smi in tox_map:
                d.tox = torch.tensor([tox_map[smi]], dtype=torch.float)
                valid_data_list.append(d)
        data_list = valid_data_list

        if not data_list:
            print("Error: No valid molecules were processed from the input data.")
            return

        train_loader, val_loader = make_loaders(data_list, test_size=0.2, batch_size=32, num_workers=num_workers)

        in_dim = data_list[0].x.shape[1]
        trainer = LocalTrainer(in_dim=in_dim, hidden=128, device="cpu", num_workers=num_workers)
        trainer.fit(train_loader, val_loader, epochs=args.epochs)

        # --- Phase 4: Generation, Evaluation & Validation ---

        # --- NEW: Filter compounds based on user criteria ---
        print(f"--- Filtering for compounds with yield > 0.5 and tox < 0.3 ---")
        filtered_df = demo[(demo['yield'] > 0.5) & (demo['tox'] < 0.3)]
        print(f"Found {len(filtered_df)} compounds matching criteria (out of {len(demo)}).")

        # We need at least 3 compounds to mix triplets
        if len(filtered_df) < 3:
            print("Warning: Not enough high-quality compounds to perform triplet mixing (need at least 3).")
            # Fallback: Use the original unfiltered list if filtering fails
            if len(known_df) < 3:
                print("Error: Not enough compounds in total to mix. Exiting.")
                return
            print("Proceeding with unfiltered list...")
            ranked_compounds = [(r.smiles, float(r.rank_score)) for _, r in known_df.iterrows()]
        else:
            ranked_compounds = [(r.smiles, float(r.rank_score)) for _, r in filtered_df.iterrows()]

        def y_pred_fn(smiles_list):
            return trainer.predict_yield(smiles_list)

        def t_pred_fn(smiles_list):
            return trainer.predict_tox(smiles_list)

        best = propose_new_compound(ranked_compounds, y_pred_fn, t_pred_fn, lam=args.lam, top_k_known=20)

        # --- VALIDATION BLOCK ---
        validation_metrics = {}
        if best["new_smiles"]:
            print("\n--- Validating Suggested Compound ---")
            validation_metrics = calculate_drug_likeness(best["new_smiles"])
            if validation_metrics:
                print(f"Lipinski's Rule of Five Passed: {validation_metrics['lipinski_rules_pass']}")
                print(f"QED (Drug-likeness) Score: {validation_metrics['qed_score']} (A score closer to 1 is better)")
                print(
                    f"Details: MW={validation_metrics['molecular_weight']}, LogP={validation_metrics['logp']}, H-Donors={validation_metrics['h_bond_donors']}, H-Acceptors={validation_metrics['h_bond_acceptors']}")
            else:
                print("Generated SMILES is invalid and could not be evaluated.")
        else:
            print("\n--- No valid new compounds were generated. ---")

        print("\n--- Final Recommendation ---")

        # --- MODIFIED: Changed source_pair to source_compounds ---
        final_output = {
            "source_compounds": best["source_compounds"],
            "new_smiles": best["new_smiles"],
            "predicted_yield": best["yield"],
            "predicted_toxicity": best["tox"],
            "objective": best["objective"]
        }
        if validation_metrics:
            final_output["validation"] = validation_metrics

        print(final_output)

    except Exception as e:
        print("An error occurred in the pipeline:")
        traceback.print_exc()


if __name__ == "__main__":
    main()