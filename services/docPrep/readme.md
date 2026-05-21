# Informationsextrahierung und Verarbeitung

# Verzeichnisstruktur

| Verzeichnis                   | Inhalt                  |
|-------------------------------|-------------------------|
| `parsed/meetings/`            | Sitzungen               |
| `parsed/persons/`             | Personen / Mandatsträger |
| `parsed/tops/`                | Tagesordnungspunkte     |
| `parsed/vorlagen/`            | Vorlage / Drucksache    |
| `attachments/[year]/[month]/` | Niederschriften & Anlagen |
| `raw_html/`                   | HTML Quelle             |

## Extrahierung von Informationen
- Für jeden Tagesordnungspunkt
  - Selektierung & Auslesung des Eintrags in der Niederschrift (pdf)
  - Zusammenfassung mit Hilfe LLM
  - Chunking & Embedding<sup>**</sup>
  - DB-ing für Vektorsuche ("Qdrant"/Rust)
    - DB Metadaten: 
      - Datum, 
      - Sitzung- und TOP-ID, 
      - Abstimmungsergebnis/-verhalten
- Für jede Anlage / Eintrag Vorlage (meist pdf)"
  - Bei Texten: Verfahren wie für TOP (s. oben)
  - Bei Graphiken: LLM Zusammenfassung

** *paraphrase-multilingual-MiniLM-L12-v2* (Lib: sentence_transformers)
