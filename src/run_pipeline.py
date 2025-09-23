import argparse
import os
import pandas as pd
from src.chembl_api import find_targets_for_disease, fetch_compounds_for_target
from src.data_processor import smiles_to_data
from src.datasets import graphs_from_csv, make_loaders
from src.train_local import LocalTrainer
from src.fragment_mix import propose_new_compound
import traceback



def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--disease_or_target", type=str, required=True,
                        help="Disease keyword (e.g., 'breast cancer') or ChEMBL target ID (e.g., CHEMBL203)")
        ap.add_argument("--is_target_id", action="store_true", help="Interpret disease_or_target as a target ChEMBL ID")
        ap.add_argument("--max_known", type=int, default=200)
        ap.add_argument("--epochs", type=int, default=3)
        ap.add_argument("--lam", type=float, default=1.0, help="tradeoff for (yield - lam * toxicity)")
        args = ap.parse_args()

        print(f"Starting pipeline with args: {args}")

        # Resolve target(s)
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

        # Fetch known compounds for the target
        known_df = fetch_compounds_for_target(chosen_target, max_compounds=args.max_known)
        if known_df.empty:
            print("No compounds found for target")
            return

        print(f"Fetched {len(known_df)} compounds")

        # Feature 1: Best-known compound
        top_row = known_df.iloc[0]
        print("Best-known compound (by pChEMBL/IC50):")
        print({
            "molecule_chembl_id": top_row.molecule_chembl_id,
            "smiles": top_row.smiles,
            "pchembl_value": top_row.pchembl_value,
            "standard_type": top_row.standard_type,
            "standard_value": top_row.standard_value,
            "units": top_row.standard_units
        })

        # Create synthetic labels for yield/toxicity for this demo
        tmp_csv = "data/known_compounds.csv"
        demo = known_df[["smiles"]].copy()
        rank = (known_df["rank_score"] - known_df["rank_score"].min()) / max(
            known_df["rank_score"].max() - known_df["rank_score"].min(), 1e-6)
        demo["yield"] = 0.5 + 0.5 * rank
        demo["tox"] = 0.5 * (1.0 - rank)
        os.makedirs("data", exist_ok=True)
        demo.to_csv(tmp_csv, index=False)
        print(f"Saved synthetic labels to {tmp_csv}")

        # Build graphs and loaders
        data_list = graphs_from_csv(tmp_csv, smiles_col="smiles", target_col="yield", default_target=0.5)
        # Attach toxicity labels to data objects
        df = pd.read_csv(tmp_csv)
        import torch
        for d, (_, r) in zip(data_list, df.iterrows()):
            d.tox = torch.tensor([float(r["tox"])], dtype=torch.float)

        train_loader, val_loader = make_loaders(data_list, test_size=0.2, batch_size=32)

        # Train multitask model locally
        in_dim = data_list[0].x.shape[1]
        trainer = LocalTrainer(in_dim=in_dim, hidden=128, device="cpu")
        trainer.fit(train_loader, val_loader, epochs=args.epochs)

        # Feature 2 & 3: Propose mix of two known compounds maximizing yield and minimizing toxicity
        ranked_pairs = [(r.smiles, float(r.rank_score)) for _, r in known_df.iterrows()]

        def y_pred_fn(smiles_list):
            return trainer.predict_yield(smiles_list)

        def t_pred_fn(smiles_list):
            return trainer.predict_tox(smiles_list)

        best = propose_new_compound(ranked_pairs, y_pred_fn, t_pred_fn, lam=args.lam, top_k_known=20)

        print("Suggested new compound from mixing two known compounds:")
        print({
            "source_compounds": best["source_pair"],
            "new_smiles": best["new_smiles"],
            "predicted_yield": best["yield"],
            "predicted_toxicity": best["tox"],
            "objective": best["objective"]
        })

    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
