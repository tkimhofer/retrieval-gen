import os.path

import pandas as pd
import requests, requests_cache
import datetime as dt
from bs4 import BeautifulSoup
from pathlib import Path
import mimetypes, hashlib
import re, json, time, random
from urllib.parse import urlparse, parse_qs, urljoin

from services.docGraph.helpers import ATTACH, VORLAGEN_DIR
from services.docGraph.helpers import read_topp, list_files, fetch_page, write_json, guess_ext, sanitize_filename, save_attachment

#### TOP -> VORLAGEN_ID und URL... beachte: nicht jeder TOP gehört zu VORLAGE

### PARAMETERISATION
BASE_URL = "https://sessionnet.owl-it.de/duisburg/bi/"
YEARS = range(2024, 2027)
CACHE_DAYS = 32
REQUEST_TIMEOUT = (10, 60*5)
SLEEP_SECONDS = 0.2

datum_pat = re.compile(r'\d{2}\.\d{2}\.\d{4}')

### setting up cache
requests_cache.install_cache(
    "bürgerportal_dui/cache/http_cache",
    expire_after=dt.timedelta(days=CACHE_DAYS)
)


tops = []
for p in list_files("bürgerportal_dui/parsed/tops/"):
    tp = read_topp(p)
    if tp['vorlage_id']:
        tops.append(read_topp(p))
len(tops)

import pandas as pd
df_full = pd.DataFrame(tops) # 59604
df_full.vorlage_id.unique().__len__() # 34073

df = df_full[['vorlage_id', 'vorlage_url']].drop_duplicates()
# df.vorlage_id.unique().__len__()
# df.shape

### loop over df, establish json and download files


def get_vorlage(url, vorlage_id):
    '''retrieve signel vorlage '''
    out = fetch_page(url, timeout=REQUEST_TIMEOUT)
    soup = BeautifulSoup(out['response'].text, "html.parser")

    table = soup.find('div', {"class": "smc-table smc-table-striped smccontenttable smc_page_vo0050_contenttable"})

    try:
        betreff = table.find('div', {'class': "smc-table-cell vobetr"}).get_text()
    except:
        betreff = soup.find("h1", {"class": "smc_h1"}).get_text(strip=True)

    vorlage_id_1 =  table.find('div', {'class': "smc-table-cell voname"}).get_text(strip=True)

    if vorlage_id_1!=vorlage_id:
        raise ValueError('vorlage id\'s stimmen nicht überein')

    aktenzeichen_ =  table.find('div', {'class': "smc-table-cell voakz"})
    aktenzeichen = aktenzeichen_.get_text(strip=True) if aktenzeichen_ else None

    typ =  table.find('div', {'class': "smc-table-cell vovaname"}).get_text(strip=True)

    docs = soup.find_all('div', {'class': "smc-el-h smc-link-normal smc_datatype_do smc-t-r991"})
    ddoc = []
    for doc in docs:
        adoc = doc.find('a')
        url = urljoin(BASE_URL, adoc['href'])
        dtype = adoc['title'].replace('Dokument Download Dateityp:', '').strip()
        dtitle = adoc.get_text()
        out = {
            'title': dtitle,
            'dtype': dtype,
            'url': url
        }
        ddoc.append(out)

    time.sleep(random.uniform(0.2, 0.7))

    # url_info = soup.find('a', {'aria-label': "Informationen"})['href']
    _url_berat = soup.find('a', {'aria-label': "Beratungen"})['href']
    url_berat = urljoin(BASE_URL, _url_berat)

    out = fetch_page(url_berat, timeout=REQUEST_TIMEOUT)
    soup_berat = BeautifulSoup(out['response'].text, "html.parser")

    accord = soup_berat.find('div', {'id': 'smcaccordion'})
    meets = accord.find_all('div', {'class': 'card card-light'})

    mm = []
    for x in meets:
        em = extract_meets(x)
        mm.append(em)

    vorlage = {
        'typ': typ,
        'betreff': betreff,
        'vorlage_id': vorlage_id_1,
        'aktenzeichen': aktenzeichen,

        'docs': ddoc,
        'url': url_berat,
        'meetings': mm,
        'datum_added': dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat() + "Z",
    }

    return vorlage

def extract_meets(meet):
    btn = meet.find('button', {'class': 'btn btn-link btn-block text-left'})#.get_text(strip=True)
    title_meet = btn.contents[0].strip()
    intent = title_meet.split('-')[-1].strip()
    datum = dt.datetime.strptime(re.match(datum_pat, title_meet).group(0), '%d.%m.%Y').date().isoformat()
    gremium = ' '.join(re.sub(datum_pat, '', title_meet).split('TOP ')[0:-1]).strip()
    ergebnis_ = meet.find('p', {'class': 'smc_field_btname'})
    ergebnis = ergebnis_.contents[1].strip() if ergebnis_ else None

    abstimmen = meet.find('p', {'class': 'smc_field_totexta'})
    stimmen = abstimmen.contents[1].strip().replace('\xa0', '') if abstimmen else None

    docs = meet.find_all('div', {'class': 'smc-el-h smc-link-normal smc_datatype_do smc-t-r991'})

    ddocs = []
    for i, doc in enumerate(docs):
        dd = doc.find('a')
        href = dd['href']
        title = dd.text.strip()
        out = {
            'title': title, 'href': urljoin(BASE_URL, href)
        }
        ddocs.append(out)

    return {'datum': datum, 'intent': intent, 'gremium' :gremium, 'title': title_meet, 'ergebnis': ergebnis, 'stimmen': stimmen, 'dateien': ddocs}


# for i in range(6302, len(df.vorlage_url)):
for i in range(len(df.vorlage_url)):  # len: 34,073

    vorlage_id = df.vorlage_id.iloc[i]
    filename = vorlage_id.replace('/', '_') + '.json'
    path = VORLAGEN_DIR / filename

    if os.path.exists(path):
        continue

    url = df.vorlage_url.iloc[i]
    vorlage_json = get_vorlage(url, vorlage_id)

    write_json(path, data=vorlage_json)
    time.sleep(SLEEP_SECONDS)
    if 0 == (i % 1000):
        print(i)



### DOWNLOAD ATTACHMENTS (DEPRECATED)

# vors = list_files("bürgerportal_dui/parsed/vorlagen/")
#
# vou = {}
# for x in vors:
#     stem = x.stem.split('_')[0]
#     if stem in vou:
#         vou[stem].append(x.stem)
#     else:
#         vou[stem] = [x.stem]
#
# len(vou) # 6215
# vou
#
# voulen = [len(d) for k,d in vou.items()]
#
# import numpy as np
# idx = np.argsort(voulen)[::-1]
# vou[list(vou)[idx[3]]]
#
#
#
# ATTACH = Path("bürgerportal_dui/attachments")
#
#
# len(vors) # 6831
# for i in range(0, 6831):
#     print(i)
#     x = vors[i]
#     vorlage = read_topp(x)
#     ndocs = 0
#     for j, d in enumerate(vorlage['docs']):
#         file_url = d['url']
#         save_attachment(file_url=file_url)
#         ndocs += 1
#         time.sleep(random.uniform(0.1, 0.3))
#
#     print(f"ndocs {ndocs}")
#     print('---')
#
