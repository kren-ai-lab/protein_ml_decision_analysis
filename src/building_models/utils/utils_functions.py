from pathlib import Path
import json
from building_models.utils.constants import (CANONICAL_RESIDUES, CANONICAL_EXTENDED_RESIDUES)
class UtilsFunctions:

    @classmethod
    def make_directory(cls, path_directory):
        Path(path_directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def export_json(cls, path_to_export, data_to_export):
        with open(path_to_export, 'w') as doc_export:
            json.dump(
                data_to_export, 
                doc_export,
                indent=4,
                default=str,
                ensure_ascii=False)
    
    @classmethod
    def checking_canonical_residues(cls, sequence:str, use_extend:bool=False):

        is_canon = True

        if use_extend:
            for residue in sequence:
                if residue not in CANONICAL_EXTENDED_RESIDUES:
                    is_canon=False
                    break
        else:
            for residue in sequence:
                if residue not in CANONICAL_RESIDUES:
                    is_canon=False
                    break
        
        return is_canon
