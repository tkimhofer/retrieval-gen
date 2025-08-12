# corpus/text

This directory contains text extracted from source PDFs** (e.g., official documents from Duisburg city council meetings – *Ratssitzung*).  
Each file is stored in JSON format, representing the structured content of a single *TOP* (Tagesordnungspunkt / agenda item).

## Folder structure
Each folder represents a source pdf file in `corpus/source_pdfs`

## File structure
- One file per TOP – file names usually reflect the meeting date and TOP number.
- Content is stored as a Python dictionary serialized to JSON, including:
  - `top_id` – identifier or agenda number
  - `drucksache_nr` – official reference/document number for the agenda item (if available)
  - `topic` – title of the agenda item  
  - `content` – extracted text of the agenda item
  - `pars` – extracted position info of agenda item (e.g. report page nb)

## Purpose
These JSON files serve as the clean, machine-readable input for:
- Downstream summarization
- Embedding and RAG (retrieval-augmented generation) pipelines  
- Analysis, search, and linking with other municipal datasets.