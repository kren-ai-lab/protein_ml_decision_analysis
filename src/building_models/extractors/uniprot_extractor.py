import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import requests
from building_models.utils.constants import BASE_URL_UNIPROT_ENTRY, TIMEOUT

class UniprotExtractor:

    @classmethod
    def fetch_uniprot_sequence(cls, uniprot_id: str) -> str:
        """
        Fetch a single UniProtKB entry sequence by accession.
        """
        url = f"{BASE_URL_UNIPROT_ENTRY}{uniprot_id}"
        headers = {"accept": "application/json"}
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data.get("sequence", {}).get("value", "") or ""
        except Exception as e:
            return ""
        
    @classmethod
    def fetch_uniprot_sequences_batch(cls, df: pd.DataFrame, id_col: str, batch_size: int = 30) -> pd.DataFrame:
        """
        Fetch sequences for a dataframe column of UniProt accessions.
        """
        rows: List[Dict[str, object]] = []
        ids = df[id_col].tolist()
        for i in tqdm(range(0, len(ids), batch_size), desc="[UniProt] Fetch", unit="batch"):
            for acc in ids[i:i+batch_size]:
                seq = cls.fetch_uniprot_sequence(acc)
                rows.append({"uniprot_id": acc, "sequence": seq})
        return pd.DataFrame(rows)