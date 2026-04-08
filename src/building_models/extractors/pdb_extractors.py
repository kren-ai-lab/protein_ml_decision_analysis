import pandas as pd
from tqdm import tqdm
from typing import Dict, Pattern, Optional, List, Iterable
from pathlib import Path
import requests
from building_models.utils.constants import BASE_URL_PDB, TIMEOUT
from Bio.PDB import PDBParser, PPBuilder

class PDBExtractor:

    @classmethod
    def download_pdb_file(cls, pdb_id: str, output_dir: Path) -> bool:
        """
        Download a single PDB file from RCSB.

        Parameters
        ----------
        pdb_id : str
            4-char PDB identifier (case-insensitive).
        output_dir : Path
            Destination directory.

        Returns
        -------
        bool
            True if downloaded successfully.
        """
        pdb_id = pdb_id.lower()
        url = f"{BASE_URL_PDB}{pdb_id}.pdb"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{pdb_id}.pdb"

        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                file_path.write_text(r.text, encoding="utf-8")
                return True
            return False
        except Exception as e:
            return False
    @classmethod
    def download_pdb_batch(cls, pdb_ids: List[str], output_dir: Path, batch_size: int = 30) -> pd.DataFrame:
        """
        Download a batch of PDB files. Returns a dataframe with statuses.
        """
        rows: List[Dict[str, object]] = []
        for i in tqdm(range(0, len(pdb_ids), batch_size), desc="[PDB] Download", unit="batch"):
            batch = pdb_ids[i:i+batch_size]
            for pid in batch:
                ok = cls.download_pdb_file(pid, output_dir)
                rows.append({"pdb_id": pid, "status_download": bool(ok)})
        return pd.DataFrame(rows)

    @classmethod
    def get_sequence_from_PDB(cls, pdb_file, chain_id, pdb_folder):

        pdb_file = f"{pdb_folder}/{pdb_file}.pdb"
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("struct", pdb_file)

        ppb = PPBuilder()
        seq = ""

        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    polypeptides = ppb.build_peptides(chain)
                    if not polypeptides:
                        continue

                    seq = "".join(str(pp.get_sequence()) for pp in polypeptides)
                    break
        return seq