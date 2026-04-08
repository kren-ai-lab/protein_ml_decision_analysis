MIN_LENGTH_SEQUENCE = 2
MAX_LENGTH_SEQUENCE = 1024

CANONICAL_RESIDUES = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                      "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

CANONICAL_EXTENDED_RESIDUES = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                      "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", 
                      "X", "U", "Z"]

COLUMNS_TO_WORK = [
    "name dataset", 
    "name source", 
    "type source", 
    "static-dynamic",	
    "license",	
    "reports constant updates", 
    "year of publication",	
    "last update date",	
    "download date",	
    "file format",	
    "protein format",	
    "category dataset",	
    "task",	
    "obtaining negative dataset",	
    "obtaining positive dataset",	
    "repository or server",	
    "publication",
    "unit of measurement"
]

BASE_URL_PDB = "https://files.rcsb.org/download/"
BASE_URL_ALPHAFOLD = "https://alphafold.ebi.ac.uk/files"
BASE_URL_UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/"
BASE_URL_UNIPARC_SEARCH = "https://rest.uniprot.org/uniparc/search"
TIMEOUT: int = 120
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

EMAIL:str = "krenai@umag.cl"