from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import time
import re

class ParsersCommons:
   
    @classmethod
    def read_fasta_doc(cls, doc_fasta, description=False):

        matrix_data = []

        for record in SeqIO.parse(doc_fasta, "fasta"):

            if description:
                row = {
                    "id" : record.id,
                    "sequence" : str(record.seq),
                    "description" : str(record.description)
                }

            else:
                row = {
                    "id" : record.id,
                    "sequence" : str(record.seq)
                }

            matrix_data.append(row)
        
        df_export = pd.DataFrame(matrix_data)
        return df_export
    
    @classmethod
    def processing_duplicated(
        cls,
        df_concat: pd.DataFrame,
        group_seq: str = "seq",
        label_col: str = "label",
        dropna_labels: bool = False,
    ):
        """
        Separate unique sequences, duplicated sequences with consistent labels,
        and duplicated sequences with conflicting annotations.

        Parameters
        ----------
        df_concat : pd.DataFrame
            Input dataframe containing sequences and annotations.
        group_seq : str, default="seq"
            Column name containing the sequence identifier.
        label_col : str, default="label"
            Column name containing the annotation/label.
        dropna_labels : bool, default=False
            Whether to ignore NaN values when assessing label conflicts.

        Returns
        -------
        df_consistent_duplicates : pd.DataFrame
            One-row-per-sequence dataframe for duplicated sequences whose labels
            are consistent across all occurrences.
        df_errors : pd.DataFrame
            Dataframe listing sequences with conflicting annotations.
        df_unique : pd.DataFrame
            Original rows corresponding to sequences that appear only once.
        """

        if group_seq not in df_concat.columns:
            raise ValueError(f"Column '{group_seq}' not found in dataframe.")
        if label_col not in df_concat.columns:
            raise ValueError(f"Column '{label_col}' not found in dataframe.")

        # Count occurrences per sequence
        seq_counts = df_concat.groupby(group_seq).size().rename("n").reset_index()

        # Unique and duplicated sequences
        unique_seqs = seq_counts.loc[seq_counts["n"] == 1, group_seq]
        duplicated_seqs = seq_counts.loc[seq_counts["n"] > 1, group_seq]

        # Keep original rows for unique sequences
        df_unique = df_concat[df_concat[group_seq].isin(unique_seqs)].copy()

        matrix_data = []
        error_sequences = []

        for sequence in duplicated_seqs:
            data_filter = df_concat[df_concat[group_seq] == sequence]

            if dropna_labels:
                labels = data_filter[label_col].dropna().unique().tolist()
            else:
                labels = data_filter[label_col].unique().tolist()

            if len(labels) == 1:
                row = {
                    group_seq: sequence,
                    label_col: labels[0],
                    "n_duplicates": len(data_filter)
                }
                matrix_data.append(row)
            else:
                error_sequences.append({
                    group_seq: sequence,
                    "labels": labels,
                    "n_duplicates": len(data_filter)
                })

        df_consistent_duplicates = pd.DataFrame(matrix_data)
        df_errors = pd.DataFrame(error_sequences)

        return df_consistent_duplicates, df_errors, df_unique
    
    @classmethod
    def read_metadata(cls, path_data, name_source, columns_to_select):
        df_metada = pd.read_excel(path_data)
        df_metada_filter = df_metada[df_metada["name source"] == name_source]
        df_metada_filter = df_metada_filter[columns_to_select]
        return df_metada_filter

    @classmethod
    def create_metadata_from_file(cls, df_metada_filter):
        dict_metadata = {}
        number_of_sources = df_metada_filter.shape[0]

        for column in df_metada_filter.columns:
            values = df_metada_filter[column].unique().tolist()

            if len(values)>1:
                values = [str(value) for value in values]
                values = ";".join(values)
                dict_metadata.update({column:values})
            else:
                dict_metadata.update({column:values[0]})

        dict_metadata.update({"number_of_sources": number_of_sources})
        dict_metadata.update({"processing_date": time.strftime("%Y-%m-%d %H:%M:%S")})
        return dict_metadata
    
    @classmethod
    def build_gene_to_sequence_dict(cls, fasta_path: str) -> dict[str, str]:
        """
        Build a mapping from gene name to protein sequence using UniProt FASTA headers.

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file.

        Returns
        -------
        dict[str, str]
            Dictionary mapping gene names to sequences.
        """
        gene_to_seq = {}

        for record in SeqIO.parse(fasta_path, "fasta"):
            header = record.description
            sequence = str(record.seq)

            match = re.search(r"\bGN=([^\s]+)", header)
            if match:
                gene_name = match.group(1).strip()
                gene_to_seq[gene_name] = sequence

        return gene_to_seq

    @classmethod
    def map_sequences_by_gene_name(
        cls,
        df: pd.DataFrame,
        fasta_path: str,
        gene_col: str = "Gene name",
        output_col: str = "sequence",
        lowercase: bool = True,
    ) -> pd.DataFrame:
        """
        Map sequences to a dataframe using gene names.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing gene names.
        fasta_path : str
            Path to the FASTA file.
        gene_col : str, default="Gene name"
            Column in df containing gene names.
        output_col : str, default="sequence"
            Name of output column with mapped sequences.
        lowercase : bool, default=True
            Whether to normalize gene names to lowercase before mapping.

        Returns
        -------
        pd.DataFrame
            Copy of dataframe with mapped sequences.
        """
        gene_to_seq = cls.build_gene_to_sequence_dict(fasta_path)

        if lowercase:
            gene_to_seq = {k.lower(): v for k, v in gene_to_seq.items()}
            keys = (
                df[gene_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        else:
            keys = df[gene_col].astype(str).str.strip()

        df_out = df.copy()
        df_out[output_col] = keys.map(gene_to_seq)

        return df_out

    @classmethod
    def report_mapping(cls, df: pd.DataFrame, seq_col: str = "sequence") -> None:
        """
        Print simple mapping statistics.
        """
        total = len(df)
        mapped = df[seq_col].notna().sum()
        unmapped = total - mapped
        print(f"Mapped:   {mapped}/{total} ({mapped / total:.2%})")
        print(f"Unmapped: {unmapped}/{total} ({unmapped / total:.2%})")