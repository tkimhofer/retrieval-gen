
import requests, requests_cache
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin
import time, re, json
import pandas as pd
import datetime as dt
from pathlib import Path
import mimetypes
import hashlib

# sys.path.append("/Users/tk/py/retrieval-gen")
from services.docGraph.infer_party import detect_parties


# defining out dirs
RAW = Path("bürgerportal_dui/raw_html")
ATTACH = Path("bürgerportal_dui/attachments")
PARSED = Path("bürgerportal_dui/parsed")
MEET_DIR = PARSED / "meetings"
TOP_DIR  = PARSED / "tops"
PERSON_DIR  = PARSED / "persons"
VORLAGEN_DIR = PARSED / "vorlagen"
# FILE_DIR = PARSED / "files"
# EDGES    = PARSED / "edges"

for p in [RAW, ATTACH, PARSED, MEET_DIR, TOP_DIR]:
    p.mkdir(parents=True, exist_ok=True)


### defining helper functions



def getSitzungsInfo(soup):
    title = soup.select_one("h1.smc_h1").get_text(" ", strip=True)

    btn = soup.select_one("#smcpanel2 h2.card-header-title button")
    location = btn.get_text(" ", strip=True) if btn else None

    info = {'title': title, 'location': location}

    # Loop all rows
    for row in soup.select("div.smc-table-row"):
        cells = row.select("div.smc-table-cell")
        if len(cells) == 2:
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(" ", strip=True)
            info[label] = value

    return info

def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def readJson(fp):
    text = fp.read_text(encoding="utf-8")
    vorlage = json.loads(text)
    return vorlage

def slugify(s: str) -> str:
    # human slug (keeps umlauts/ß), for gremium etc.
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w-]", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def fetch_page(url: str, cache_expiry=None, timeout=(10, 30)):
    kwargs = {"timeout": timeout}

    if cache_expiry is not None:
        kwargs["expire_after"] = cache_expiry

    res = requests.get(url, **kwargs)
    res.raise_for_status()
    res.encoding = "utf-8"

    return {
        "source_url": url,
        "fetched_at": now_iso(),
        "from_cache": getattr(res, "from_cache", False),
        "status": res.status_code,
        "response": res,
    }

def save_raw(RAW, page_slug, entity_id, text):

    fn = RAW / f"{dt.date.today()}_{page_slug}_{entity_id}.html"
    fn.write_text(text, encoding="utf-8")

    # 3. Return the file path as a string
    return str(fn)

def extract_meeting_urls(url, BASE_URL):
    ### extract urls of meetings (sitzung) from a commitee (gremium) page

    out = fetch_page(url, cache_expiry=-1)
    soup = BeautifulSoup(out['response'].text, "html.parser")

    table = soup.find("table")
    if not table:
        return []
    rows = table.find_all("tr")[1:]

    urls = []
    for row in rows:
        test=row.find_all('a')
        if len(test)>1:
            href = test[1]['href']
            if 'si0057' in href:
                urls.append(BASE_URL + test[1]['href'])

    return urls

def get_url_info(url):
    p = urlparse(url)
    path_comp = p.path.split("/")
    page =  path_comp[-1].replace(".asp", "")
    return p._asdict() | {"page": page,}

def extract_meeting_info(soup):
    # title = soup.select_one("h1.smc_h1").get_text(" ", strip=True)
    h1 = soup.select_one("h1.smc_h1")
    title = h1.get_text(" ", strip=True) if h1 else None

    btn = soup.select_one("#smcpanel2 h2.card-header-title button")
    location = btn.get_text(" ", strip=True) if btn else None

    info = {'title': title, 'location': location}

    # Loop all rows
    for row in soup.select("div.smc-table-row"):
        cells = row.select("div.smc-table-cell")
        if len(cells) == 2:
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(" ", strip=True)
            info[label] = value

    return info

def save_meeting(meeting: dict, BASE_URL: str) -> Path:
    """
    meeting: {
      meeting_id:int, title:str, gremium:str, date:str(YYYY-MM-DD),
      time:str(HH:MM[:SS][±TZ]) or None, datetime:str(ISO) or None,
      location:str|None, source_url:str
    }
    """
    # ensure slug + ISO datetime
    meeting = dict(meeting)  # copy
    # meeting["gremium_slug"] = meeting.get("gremium_slug") or slug(meeting["Gremium"])
    meeting["saved_at"] = now_iso()
    out = MEET_DIR / f'''{meeting["Sitzung"].replace('/', '_')}.json'''
    write_json(out, meeting)
    return out

def getTopFiles(tr, BASE_URL: str):
    td_docs = tr.find_all('td', recursive=False)
    files, seen = [], set()
    if len(td_docs) > 3:
        for card in td_docs[3].select(".smc-doc-content"):
            text_link = card.select_one('.smc-text-block-991 a[href*="getfile.asp"]') \
                        or card.select_one('a[href*="getfile.asp"]')
            if not text_link:
                continue
            abs_url = urljoin(BASE_URL, text_link["href"].strip())
            q = parse_qs(urlparse(abs_url).query)
            doc_id = q.get("id", [abs_url])[0]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            label = (text_link.get_text(" ", strip=True)
                     or text_link.get("title", "")
                     or text_link.get("aria-label", "")
                     or "Dokument")
            files.append({
                "url": abs_url,
                "label": label,  # e.g. "Beschlussvorlage", "Einbahnstraßenkonzept"
                "id": q.get("id", [None])[0],
                "type": q.get("type", [None])[0],
            })
    return files

def extract_anwesende(url, BASE_URL: str) -> list[dict]:

    # matching url
    url_anwesenheit = BASE_URL + "to0045.asp?__ksinr=" + url.split('ksinr=')[-1]

    out = fetch_page(url_anwesenheit, cache_expiry=-1)

    sout = out['response']
    soup = BeautifulSoup(sout.text, "html.parser")

    table = soup.select_one("#smc_page_to0045_contenttable1")  # adjust if your id differs
    if not table:
        raise ValueError('Keine Anwesenheitsliste gefunden')

    members = []
    rows = table.select("tr")
    for tr in rows:
        cells = tr.find_all('td')
        if len(cells) < 2:
            continue

        # name und interne id
        cell = cells[0]
        a = cell.find("a")
        name = a.get_text(strip=True) if a else cell.get_text(strip=True)

        href = a.get("href") if a else None
        id = href.split('__kpenr=')[-1].split('&')[0] if href else None

        # 2nd cell: party (if present)
        party = cells[1].get_text(strip=True) if len(cells) > 1 else None

        # 3rd cell: role (fallback to text if class not present)
        role = cells[2].get_text(strip=True) if len(cells) > 2 else None

        # 4th cell: extra/notes (may be empty)
        extra = cells[3].get_text(strip=True) if len(cells) > 3 else None


        member = {
            "name": name or None,
            "id": id,
            "party": party or None,
            "role": role or None,
            "extra": extra or None,
        }
        members.append(member)

    return members

def extract_tops_and_files(soup, BASE_URL: str, meeting_id:str) -> list[dict]:
    # soup = BeautifulSoup(html, "lxml")  # or "html.parser"
    #
    table = soup.select_one("#smc_page_si0057_contenttable2")  # adjust if your id differs
    if not table:
        return []

    tops = []
    type = None
    rows = table.select("tr")
    for tr in rows:

        beratungsergebnis_1 = None
        beratungsergebnis_2 = None
        stimmenverteilung = None
        antrag = None

        badge = tr.select_one('td.tofnum span.badge')
        label = badge.get_text(strip=True) if badge else None  # -> "Ö 1"
        m = re.match(r'([ÖN])\s*(\d+(?:\.\d+)*)', label or "")

        if m:
            section_code = m.group(1)  # "Ö" (öffentlich) or "N" (nichtöffentlich)
            top_nr = m.group(2)  # e.g. "1", "3.2"
            # section = {'Ö': 'öffentlich', 'N': 'nichtöffentlich'}.get(section_code, section_code)
            if section_code == 'N': continue
            title_div = tr.select_one(".smc-card-header-title-simple")
            top_name = title_div.get_text(" ", strip=True)


            beschluss = tr.select_one("p", {'class': "smc_field_smcdv0_box2_beschluss margin-bottom-0 margin-top-0"})
            if beschluss:
                beratungsergebnis_1 = beschluss.get_text().replace('\xa0', '')

            abstimmung = tr.select("p", {'class': "smc_field_smcdv0_box2_abstimmung margin-bottom-0 margin-top-0"})
            if len(abstimmung) > 0:
                beratungsergebnis_2 = abstimmung[0].get_text().replace('\xa0', '') if abstimmung else None

                if (len(abstimmung) > 1):
                    stimmenverteilung = abstimmung[-1].get_text().replace('\xa0', '')


            # if (type == 'Beschlussvorlagen') | (type == "Anträge/Anfragen"):
            #
            #     beschluss = tr.select_one("p", {'class': "smc_field_smcdv0_box2_beschluss margin-bottom-0 margin-top-0"})
            #     beratungsergebnis_1 = beschluss.get_text().replace('\xa0', '')
            #
            #     abstimmung = tr.select("p", {'class': "smc_field_smcdv0_box2_abstimmung margin-bottom-0 margin-top-0"})
            #
            #     beratungsergebnis_2 = abstimmung[0].get_text().replace('\xa0', '') if abstimmung else None
            #
            #     if abstimmung and (len(abstimmung)>1):
            #         stimmenverteilung = abstimmung[-1].get_text().replace('\xa0', '')
            #     else:
            #         stimmenverteilung = None

            beschl = {
                'beratungsergebnis_1': beratungsergebnis_1,
                'beratungsergebnis_2': beratungsergebnis_2,
                'stimmenverteilung': stimmenverteilung,
            }


            if (type != None) and ( (type == "Anträge/Anfragen") or ('Anfrage' in type) or ('Antrag' in type) ):
                # print(top_name)
                # test.append(top_name)
                antrag = detect_parties(top_name)
                # print(f"""{section}: {top_nr} {"(" + type + ")" if type else ''}...{top_name}""")

            vorlage_a = (
                    tr.select_one('[data-label="Vorlage"] a.smc_datatype_vo') or
                    tr.select_one('a.smc_datatype_vo') or
                    tr.select_one('[data-label="Vorlage"] a[href*="vo"]') or
                    tr.select_one('a[href*="vo0050.asp"]')
            )
            vorlage_id = None
            vorlage_url = None
            if vorlage_a and vorlage_a.has_attr("href"):
                vorlage_id = vorlage_a.get_text(" ", strip=True)
                vorlage_url = urljoin(BASE_URL, vorlage_a["href"])

            files = getTopFiles(tr)

            # if m.string == "Ö 6":
            #     print('check top')
            #     break

            top_info = {
                'top_id': ":".join([meeting_id, 'Ö', top_nr]),
                'meeting_id': meeting_id,
                # 'top_nr': top_nr,
                'top_name': top_name,
                'vorlage_id': vorlage_id,
                'vorlage_url': vorlage_url,
                'antrag': antrag,
                'beschluss': beschl,
                'files': files,
                'type': type
            }

            tops.append(top_info)


            ### check vorlage url and see what has been decided in which meetings etc




        else:
            # print(label)
            # asdf
            # print(label)
            title_div = tr.select_one(".smc-card-header-title-simple")
            if title_div:
                top_name = title_div.get_text(" ", strip=True)
                type = top_name
                print(type)
                # if 'nichtöffentliche' not in top_name:
                #     type = top_name
                # else:
                #     continue
                # top_name = title_div.get_text(" ", strip=True)
                # print(top_name)
            else:
                continue

    return tops

def save_top(top: dict) -> Path:
    """
    top: {
      meeting_id:int, nr_raw:str, title:str, aktenzeichen:str|None,
      vorlage_id:int|None, vorlage_url:str|None,
      files:list[{file_id:int|str,label,url,path}]
    }
    """
    nr = top['top_id'].replace('/', '_')
    top_id = nr
    rec = {
        "top_id": top_id,
        "meeting_id": top['meeting_id'],
        # "nr_raw": top["nr_raw"],
        # "nr": nr,
        "name": top["top_name"],
        "public": True,
        'type': top['type'],
        "antrag_party": list(top.get("antrag")) if top.get("antrag") else [],

        "aktenzeichen": top.get("aktenzeichen"),
        "vorlage_id": top.get("vorlage_id"),
        "vorlage_url": top.get("vorlage_url"),

        "beschluss": top.get("beschluss"),
        "files": top.get("files", []),
        "saved_at": now_iso(),
    }
    # persist TOP JSON
    out = TOP_DIR / f'{top_id}.json'
    write_json(out, rec)

    return out

def read_topp(file_path:Path):
    text = file_path.read_text(encoding="utf-8")
    return json.loads(text)

def clean(txt):
    return " ".join(txt.get_text(" ", strip=True).split()) if txt else None

def extract_css_email_parts(soup):
    """Extract CSS ::before / ::after email fragments."""
    css = "\n".join(style.get_text() for style in soup.find_all("style"))

    parts = {}

    pattern = re.compile(
        r"span\.(?P<class>[^\s:{]+)::(?P<pseudo>before|after)\s*"
        r"\{\s*content:\s*\"(?P<content>[^\"]*)\"",
        re.I,
    )

    for m in pattern.finditer(css):
        cls = m.group("class")
        pseudo = m.group("pseudo")
        content = m.group("content")

        parts.setdefault(cls, {})[pseudo] = content

    return parts

def decode_email(cell, css_email_parts):
    span = cell.find("span")
    if not span:
        return clean(cell)

    classes = span.get("class", [])
    visible = span.get_text(strip=True)

    before = ""
    after = ""

    for cls in classes:
        if cls in css_email_parts:
            before = css_email_parts[cls].get("before", before)
            after = css_email_parts[cls].get("after", after)

    return after + visible + before[::-1]

def get_person_name(x_str):
    out = x_str.split(',')

    anrede_name = out[0].split()

    return {
        'anrede': anrede_name[0].strip(),
        'vorname': out[1],
        'nachname': ' '.join([x.strip() for x in anrede_name[1:]]),
     }

def guess_ext(res: requests.Response, url: str) -> str:
    ct = (res.headers.get("Content-Type") or "").split(";")[0].lower()
    path_ext = Path(urlparse(url).path).suffix
    if ct == "application/pdf" or url.lower().endswith(".pdf"): return ".pdf"
    if "msword" in ct or path_ext == ".doc": return ".doc"
    if "officedocument.wordprocessingml.document" in ct or path_ext == ".docx": return ".docx"
    if path_ext: return path_ext
    return mimetypes.guess_extension(ct) or ".bin"

def sanitize_filename(name: str) -> str:
    # keep umlauts/ß; just remove illegal path chars and tidy spaces
    name = name.replace("\u00A0", " ").strip()
    name = re.sub(r'[\\/:*?"<>|]+', '-', name)  # Windows-forbidden
    name = re.sub(r'\s+', ' ', name)
    return name

def save_attachment(file_url: str, label: str | None = None,
                    cache_expiry=None, timeout=(10, 120),
                    extra_meta: dict | None = None) -> dict:
    res = requests.get(file_url, timeout=timeout, expire_after=cache_expiry)
    res.raise_for_status()

    q = parse_qs(urlparse(file_url).query)
    doc_id  = (q.get("id") or [None])[0]
    doc_type= (q.get("type") or [None])[0]
    ext = guess_ext(res, file_url)

    y, m = dt.date.today().year, dt.date.today().month
    outdir = ATTACH / f"{y:04d}" / f"{m:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # filename: <id>_<Label>.pdf (preserve umlauts/ß)
    if not doc_id:
        doc_id = hashlib.sha1(file_url.encode("utf-8")).hexdigest()[:12]
    pretty = sanitize_filename(label or "Dokument")
    fpath = outdir / f"{doc_id}_{pretty}{ext}"

    wrote = False
    if not fpath.exists():
        fpath.write_bytes(res.content)
        wrote = True

    meta = {
        "source_url": file_url,
        "saved_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat() + "Z",
        "from_cache": getattr(res, "from_cache", False),
        "status": res.status_code,
        "content_type": res.headers.get("Content-Type"),
        "content_length": res.headers.get("Content-Length"),
        "id": doc_id,
        "type": doc_type,
        "label": label,
        "path": str(fpath),
        **(extra_meta or {}),
    }
    with open(f"{fpath}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    return {"path": str(fpath), "wrote": wrote, "meta": meta}

def list_files(directory: str):
    path = Path(directory)
    return [p for p in path.iterdir() if p.is_file()]


def extract_files_from_td(td, base_url):
    files, seen = [], set()

    # iterate per doc "card"
    for card in td.select('.smc-doc-content, .smc-doc, .smc-documents .smc-doc-ds-1'):
        # prefer the *text* link, not the blue button
        text_link = (
            card.select_one('.smc-text-block-991 a[href*="getfile.asp"]')
            or card.select_one('.smc-text-block a[href*="getfile.asp"]')
            or card.select_one('a[href*="getfile.asp"]')
        )
        if not text_link:
            continue

        abs_url = urljoin(base_url, text_link['href'].strip())
        q = parse_qs(urlparse(abs_url).query)
        doc_id = q.get('id', [abs_url])[0]
        if doc_id in seen:
            continue
        seen.add(doc_id)

        label = (
            text_link.get_text(' ', strip=True)
            or text_link.get('title', '')
            or text_link.get('aria-label', '')
            or 'Dokument'
        )
        files.append({
            'url': abs_url,
            'label': label,                     # <-- "Beschlussvorlage", "Einbahnstraßenkonzept"
            'id': q.get('id', [None])[0],
            'type': q.get('type', [None])[0],
        })
    return files


def scrape_drucksache(detail_url):
    resp = requests.get(detail_url)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    # Example: info is in a <table> with rows
    info = {}
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) == 2:
            key = cols[0].get_text(strip=True)
            val = cols[1].get_text(strip=True)
            info[key] = val

    # PDF link
    pdf_link = None
    pdf_tag = soup.find("a", href=lambda href: href and href.endswith(".pdf"))
    if pdf_tag:
        pdf_link = f"https://sessionnet.owl-it.de/duisburg/bi/{pdf_tag['href']}"

    info["pdf_link"] = pdf_link
    info["detail_url"] = detail_url
    return info


### DATEN

tt = {
    'Sozialdemokratischen Partei Deutschlands': ['SPD'],
    'Christlich-Demokratische Union': ['CDU'],
    'Bündnis 90/Die Grünen': ['Bündnis 90/Die Grünen','Bündis 90/Die Grünen', 'Bündnis 90/ Die Grünen', 'Bündis 90/Die Grünen ', 'Bündnis 90 /Die Grünen'],
    'Alternative für Deutschland': ['AfD'],
    'Junges Duisburg': ['Junges Duisburg', 'JUDU'],
    'Duisburger Alternative Liste': ['DAL'],
    'Freie Demokratische Partei': ['FDP'],
    'Fraktion Die Linke./Die PARTEI': ['Die Linke./Die PARTEI', 'Die Linke./ Die PARTEI'],
    'Bündnis Sahra Wagenknecht': ['Bündnis Sahra Wagenknecht'],
    'Sozial, Gerecht, Unabhängig': ['SGU'],
    'Die Linke.': ['Die Linke.'],
    'parteilos': ['parteilos', 'Parteilos'],
    'Fraktion Tierschutz/DAL': ['Tierschutz/DAL'],
    'Solidarität für Duisburg': ['SfD', 'SFD'],
    'Tierschutz': ['Tierschutz'],
    'Wir gestalten Duisburg': ['WGD'],
    'Fraktion TR/DAL': ['TR/DAL', 'TR / DAL'],
    'Aktive Bürgerinitiative': ['ABI'],
    'Müslüman Türkler Birligi': ['MTB'],
    'Tükische Repräsentanz': ['TR'],
    'Bündinis für Innovations und Gerechtigkeit': ['BIG-DERGAH'],
    'AG-Handicap': ['AG-Handicap'],
    'AG-Handicap / VKM-Duisburg e.V.': ['AG-Handicap / VKM-Duisburg e.V.'],
    'Bürgerlich-Liberale': ['BL'],
    'Die PARTEI': ['Die PARTEI']
}

xmap = {}
for k, d in tt.items():
    for y in d:
        xmap[y] = k


partei_map = {
    'Sozialdemokratischen Partei Deutschlands': {'handle': 'SPD', 'typ': 'partei', 'url': 'https://www.spd-duisburg.de/'},
    'Christlich-Demokratische Union': {'handle': 'CDU', 'typ': 'partei', 'url': 'https://www.cdu-duisburg.de/'},
    'Bündnis 90/Die Grünen': {'handle': 'Grüne', 'typ': 'partei', 'url': 'https://gruene-duisburg.de/'},
    'Alternative für Deutschland': {'handle': 'AfD', 'typ': 'partei', 'url': 'https://afd-duisburg.de/'},
    'Junges Duisburg': {'handle': 'JUDU', 'typ': 'wählergemeinschaft', 'url': 'https://junges-duisburg.de/'},
    'Duisburger Alternative Liste':  {'handle': 'DAL', 'typ': 'wählergemeinschaft', 'url': 'https://www.dal-politik.de/'},
    'Freie Demokratische Partei': {'handle': 'FDP', 'typ': 'partei', 'url': 'https://fdp-duisburg.de/'},
    'Die Linke.': {'handle': 'Linke', 'typ': 'partei', 'url': 'https://www.dielinke-duisburg.de/'},
    'Die PARTEI':  {'handle': 'Partei', 'typ': 'partei', 'url': 'https://www.die-partei-duisburg.de/'},
    'Fraktion Die Linke./Die PARTEI':  {'handle': 'linke/partei', 'typ': 'fraktion'},
    'Bündnis Sahra Wagenknecht': {'handle': 'BSW', 'typ': 'partei', 'url': 'https://www.fraktion-bsw-duisburg.de/'},
    'Sozial, Gerecht, Unabhängig':  {'handle': 'SGU', 'typ': 'wählergemeinschaft', 'url': 'https://www.sgu-duisburg.de/'},
    'parteilos': {'handle': 'parteilos', 'typ': 'parteilos', 'url':None},
    'Fraktion Tierschutz/DAL': {'handle': 'Fraktion Ts/DAL', 'typ': 'fraktion', 'url': 'https://tierschutz-hier.de/ratsfraktion-tierschutz-dal-duisburg'},
    'Solidarität für Duisburg': {'handle': 'SfD', 'typ': 'wählergemeinschaft', 'url':None},
    'Wir gestalten Duisburg': {'handle': 'WGD', 'typ': 'wählergemeinschaft', 'url': 'http://dalduisburg.de/'},
    # 'TR/DAL': ['TR/DAL'],
    'Aktive Bürgerinitiative': {'handle': 'ABI', 'typ': 'wählergemeinschaft', 'url': 'https://abi-für-duisburg.de/'},
    'Müslüman Türkler Birligi': {'handle': 'MTB', 'typ': 'wählergemeinschaft', 'url':None},
    'Tükische Repräsentanz': {'handle': 'TR', 'typ': 'wählergemeinschaft', 'url':None},
    'Fraktion TR/DAL': {'handle': 'TR/DAL', 'typ': 'fraktion', 'url':None},
    'Bündinis für Innovations und Gerechtigkeit': {'handle': 'BIG-DERGAH', 'typ': 'fraktioin', 'url':None},
    'AG-Handicap': {'handle': 'AG-Handicap', 'typ': 'wählergemeinschaft', 'url':None},
    'AG-Handicap / VKM-Duisburg e.V.': {'handle': 'AG-Handicap', 'typ': 'wählergemeinschaft', 'url':None},
    'Bürgerlich-Liberale': {'handle':'BL', 'typ': 'wählergemeinschaft', 'url':None},
    'Tierschutz': {'handle': 'Tierschutz', 'typ': 'partei', 'url': 'https://tierschutz-hier.de'},
}

