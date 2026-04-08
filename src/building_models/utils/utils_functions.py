from pathlib import Path
import json

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