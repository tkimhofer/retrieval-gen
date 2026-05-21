import requests_cache
import datetime as dt
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from services.docGraph.helpers import MEET_DIR
from services.docGraph.helpers import readJson, fetch_page, save_attachment


#### NIEDERSCHRIFTEN BEFINDEN SICH UNTER DOKUMENTE-LISTE AUF SITZUNGS-SEITE

### PARAMETERISATION
BASE_URL = "https://sessionnet.owl-it.de/duisburg/bi/"
YEARS = range(2024, 2027)
CACHE_DAYS = 32
REQUEST_TIMEOUT = (10, 60*5)
SLEEP_SECONDS = 0.5

### setting up cache
requests_cache.install_cache(
    "bürgerportal_dui/cache/http_cache",
    expire_after=dt.timedelta(days=31)  # default: 31 days
)

#
# def getRequest(url, cache_expiry:dt.timedelta=None, timeout=(10, 30)):
#     ### timeout: 2-tuple with connection and then read timeout
#
#     if cache_expiry:
#         res = requests.get(url, timeout=timeout, expire_after = cache_expiry)
#     else:
#         res = requests.get(url, timeout=timeout)
#
#     res.encoding = "utf-8"
#
#
#     return {
#         'source_url': url,
#         'datum': dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat() + "Z",
#         "from_cache": getattr(res, "from_cache", False),
#         "status": res.status_code,
#         'response': res,
#     }

fh_meets = [
   p for p in MEET_DIR.iterdir()
    if (
        p.is_file()
        and p.suffix == ".json"
        and len(p.stem.split("_")) == 3
        and int(p.stem.split("_")[1]) in YEARS
        # and 'RAT' in p.stem
    )
]

href = {}
not_found = {}
vorlagen = {}

for i in range(0, len(fh_meets)):

    fp = fh_meets[i]
    vor = readJson(fp)
    fname = fp.name.replace('.json', '')

    url_meet = vor['source_url']
    out = fetch_page(url_meet, timeout=REQUEST_TIMEOUT)
    soup = BeautifulSoup(out['response'].text, "html.parser")

    docz = soup.find_all("div", {'class': 'smc-el-h smc-link-normal smc_datatype_do smc-t-r991'})

    if 'abgesagt' in vor['title'].lower():
        href[fname] = 'abgesagt'
        continue

    ergebnis = False
    for k, dcl in enumerate(docz):

        a = dcl.find('a')
        title = a.text
        # print(title)
        if 'ergebnisprotokoll' in title.lower():
            ergebnis = True
            print(title)

        elif 'niederschrift'  in title.lower():
            href[fname] = a['href']
            # print('done')
            file_url = urljoin(BASE_URL, a['href'])
            save_attachment(file_url, label = fname)
            # print('got it')
            print(title)
            continue
    #
    # if fname not in href:
    #     if ergebnis:
    #         if 'ergebnisprotokoll' in title.lower():
    #             href[fname] = a['href']
    #             # print('done')
    #             file_url = urljoin(BASE_URL, a['href'])
    #             save_attachment(file_url, label=fname)
    #             # print('got it')
    #             continue
    #     else:
    #         not_found[fname] = url_meet
    #         # print(vor['title'])
    #         # print(url_meet)



