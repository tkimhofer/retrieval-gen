import requests_cache
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin
import datetime as dt

from services.docGraph.helpers import RAW, PERSON_DIR
from services.docGraph.helpers import write_json, fetch_page, save_raw,  clean, get_person_name
from services.docGraph.helpers import xmap, partei_map


#### DISCOVERY & CRAWLING STRATEGIE:
# KOMMUNALPERSON LISTE -> PERSON DETAILS (z.B NAME, MANDATSTRÄGER, PARTEI)

### PARAMETERISATION
BASE_URL = "https://sessionnet.owl-it.de/duisburg/bi/"
YEARS = range(2002, 2027)
CACHE_DAYS = 1
REQUEST_TIMEOUT = (10, 60*5)
SLEEP_SECONDS = 0.5

### BEISPIEL OUTPUT FÜR PERSON (JSON)
# {
#   "anrede": "Herr",
#   "vorname": "Sören",
#   "nachname": "Oberbürgermeister Link",
#   "partei": null,
#   "href": "https://sessionnet.owl-it.de/duisburg/bi/pe0051.asp?__kpenr=5717",
#   "person_id": "5717",
#   "data": {
#     "Ort": "47049 Duisburg",
#     "Straße": "Burgplatz 19",
#     "Telefon dienstl.": "0203-283 2106",
#     "Fax dienstl.": "0203-283 3976",
#     "E-Mail": "kni"
#   },
#   "ende": null,
#   "beginn ende": null
# }

### setting up cache
requests_cache.install_cache(
    "bürgerportal_dui/cache/http_cache",
    expire_after=dt.timedelta(days=CACHE_DAYS)
)

### URL von mandatsträgern
url = urljoin(BASE_URL, "kp0041.asp")
out = fetch_page(url, timeout=REQUEST_TIMEOUT)
save_raw(RAW, page_slug='mandatsträger', entity_id='kp0041',  text=out['response'].text)

soup = BeautifulSoup(out['response'].text, "html.parser")

table = soup.find("table")
rows = table.find_all("tr")[1:]

p = []
for row in rows:
    rcols = row.find_all('td')
    if not rcols: continue

    if len(rcols) != 4:
        print(rcols)

    person = {}
    col_name = row.find('td', {"data-label": "Name"})
    name_str = col_name.get_text(strip=True)

    if name_str.lower() in ['n.n', 'n.n.', 'nn']:
        continue

    name = get_person_name(name_str)

    person['anrede'] = name['anrede']
    person['vorname'] = name['vorname'].strip()
    person['nachname'] =  name['nachname'].strip()

    col_partei = row.find('td', {"data-label": "Mitgliedschaft"})
    partei = col_partei.get_text(strip=True) if col_partei else None

    if partei != '':
        try:
            pp = xmap[partei]
            person['partei'] = partei_map[pp]
        except:
            print(person)
            print(partei)
            break
    else:
        person['partei'] = None

    try:
        a = row.find("a", href=True)
        if not a:
            person["href"] = None
            person["person_id"] = None
        else:
            url_ext = a["href"]
            # href = base_url + url_ext
            href = urljoin(BASE_URL, url_ext)
            person['href'] = href
            query = urlparse(href).query
            person['person_id'] = parse_qs(query).get("__kpenr", [None])[0]
    except:
        person['href']= None
        person['person_id'] = None
        continue

    try:
        per_url = fetch_page(href, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(per_url['response'].text, "html.parser")
        data = {}
        for detail_row in soup.select(".smc-table-row"):
            label_cell = detail_row.select_one(".smc-cell-head")
            value_cell = detail_row.select_one(".smc-table-cell:not(.smc-cell-head)")

            if not label_cell or not value_cell:
                continue

            key = clean(label_cell)

            if key == "Internet":
                a = value_cell.select_one("a[href]")
                value = a["href"].strip() if a else clean(value_cell)
            else:
                value = clean(value_cell)

            if not value:
                continue

            if key in data:
                if not isinstance(data[key], list):
                    data[key] = [data[key]]
                data[key].append(value)
            else:
                data[key] = value

        person["data"] = data
    except:
        person['data'] = {}

    col_ende = row.find('td', {"data-label": "Ende"})
    person['ende'] = col_ende.get_text(strip=True)
    person['ende'] = None if person['ende'] == '' else person['ende']

    col_zeit = row.find('td', {"data-label": "Beginn Ende"})
    person['beginn ende'] = col_zeit.get_text(strip=True)
    person['beginn ende'] = None if person['beginn ende'] == '' else person['beginn ende']

    path = PERSON_DIR / f"{person['person_id']}.json"

    write_json(path, person)

    p.append(person)


# import pandas as pd
# personen=pd.DataFrame(p)
# personen.partei.value_counts()
