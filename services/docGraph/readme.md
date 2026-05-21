# Bürgerportal des Ratsinformationssystems

## SessionNet
- Gremieninformationssystem basiert auf Technologie *Active Server Pages (.asp)*

| URL Muster | Bedeutung |
|---|---|
| `gr[xxxx].asp` | Gremien |
| `si[xxxx].asp` | Sitzungen |
| `kp[xxxx].asp` | Personen / Mandatsträger |
| `pe[xxxx].asp` | Person detail |
| `vo[xxxx].asp` | Vorlagen |
| `do[xxxx].asp` | Dokumente |


## Daten- und Dokumentenerschließung
1. Gremienliste (gr[xxxx].asp)
   - Für jedes Gremium:
     - Liste von Sitzungen
     - Für jede Sitzungen:
       - Teilnehmer
       - Tagesordnungspunkte (Vorlage, Dateien)
       - Niederschrift mit Positionen und ggf Abstimmungsverhalten
2. Kommualperson-Liste (kp[xxxx].asp):
    - Für jede Kommunalperson:
        - Funktion / Mandat
        - Parteizugehörigkeit

## Download Details
- Caching: 
  - 31d (und teilw. ungegrenzt) für Gremien & Sitzungen
  - 1d für Kommunalpersonen
