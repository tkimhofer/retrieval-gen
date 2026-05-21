#### RESOURCE DISCOVERY & CRAWLING
import requests_cache
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import pandas as pd
import datetime as dt

from services.docGraph.helpers import fetch_page, save_raw, extract_meeting_urls, get_url_info, extract_meeting_info, save_meeting, extract_anwesende, extract_tops_and_files, save_top


#### DISCOVERY & CRAWLING STRATEGIE:
# GREMIEN LISTE -> GREMIUM -> SITZUNGS LISTE -> SITZUNG -> TEILNEHMER LISTE, VORLAGEN, TAGESORDNUNGSPUNKTE, ANLAGEN
#  3,122 sitzungen
# 69,598 tops


### PARAMETERISATION
BASE_URL = "https://sessionnet.owl-it.de/duisburg/bi/"
YEARS = range(2002, 2027)
CACHE_DAYS = 31
REQUEST_TIMEOUT = (10, 60)
SLEEP_SECONDS = 0.5


### BEISPIEL OUTPUT FÜR SITZUNG (JSON)
# {
#   "title": "öffentliche/nichtöffentliche Sitzung des Rates der Stadt - 24.02.2026 - 15:00-20:20 Uhr",
#   "location": "Sitzungsort",
#   "Sitzung": "RAT/2026/0001",
#   "Gremium": "Rat der Stadt",
#   "Datum": "24.02.2026",
#   "Zeit": "15:00-20:20 Uhr",
#   "source_url": "https://sessionnet.owl-it.de/duisburg/bi/si0057.asp?__ksinr=20094920",
#   "attendence": [
#     {
#       "name": "Herr Oberbürgermeister Link, Sören",
#       "id": "5717",
#       "party": "SPD",
#       "role": "Oberbürgermeister",
#     },
#     {
#       "name": "Ratsherr Baser, Ünsal",
#       "id": "6321",
#       "party": "SPD",
#       "role": "Ratsmitglied",
#     },
#     {...}]


#### BEISPIEL OUTPUT FÜR TAGESORDNUNGSPUNKT (TOP)
# {
#   "top_id": "RAT_2026_0001:Ö:52",
#   "meeting_id": "RAT/2026/0001",
#   "name": "Der ökonomische Impact der Logistikunternehmen in Duisburg: Ergebnisse der Studie vom Institut der deutschen Wirtschaft für den Wirtschaftsstandort Duisburg",
#   "public": true,
#   "type": "Mitteilungsvorlagen",
#   "antrag_party": [],
#   "aktenzeichen": null,
#   "vorlage_id": "25-1417",
#   "vorlage_url": "https://sessionnet.owl-it.de/duisburg/bi/vo0050.asp?__kvonr=20133363",
#   "beschluss": {
#     "beraturngsergebnis_1": "Beratungsergebnis: Kenntnis genommen",
#     "beraturngsergebnis_2": "Beratungsergebnis: Kenntnis genommen",
#     "stimmenverteilung": null
#   },
#   "files": [
#     {
#       "url": "https://sessionnet.owl-it.de/duisburg/bi/getfile.asp?id=1775217&type=do",
#       "label": "Mitteilungsvorlage",
#       "id": "1775217",
#       "type": "do"
#     },
#     {
#       "url": "https://sessionnet.owl-it.de/duisburg/bi/getfile.asp?id=1781721&type=do",
#       "label": "Anlage DS 25-1417 Logistikstudie",
#       "id": "1781721",
#       "type": "do"
#     }
#   ],
#   "saved_at": "2026-05-19T16:55:01Z"
# }

### setting up cache
requests_cache.install_cache(
    "bürgerportal_dui/cache/http_cache",
    expire_after=dt.timedelta(days=CACHE_DAYS)
)

### DISCOVER "GREMIEN"
page_slug = 'gremien'
gremien_url = urljoin(BASE_URL, "gr0040.asp")

out = fetch_page(gremien_url, cache_expiry=-1)
page_id = get_url_info(gremien_url)['page']
save_raw(page_slug, page_id , out['response'].text)

soup = BeautifulSoup(out['response'].text, "html.parser")

table = soup.find("table")
rows = table.find_all("tr")[1:]


### SITZUNGEN
meetings = []
yrs = list(range(2000, 2027, 1))

gr = ''
for row in rows:
    rtd = row.find_all('td')
    gremium = rtd[0].find('a').get_text(strip=True) # gremium name
    if gr != gremium:
        # swap gremium
        gr = gremium

    # go to gremium page, find "sitzungen"
    url_gremium = BASE_URL + rtd[0].find('a')['href'] # grnr gremium-nummer
    out_gr = fetch_page(url_gremium, cache_expiry=-1)
    soup = BeautifulSoup(out_gr['response'].text, "html.parser")

    url_sitzung = soup.find('a', {'aria-label': 'Sitzungen'})
    if url_sitzung:
        url_sitzung = url_sitzung['href']
    else:
        continue

    ### sitzungen last 12m or so... using year as search par instead (see further below)
    url_use = BASE_URL + url_sitzung
    out = fetch_page(url_use, cache_expiry=-1)
    soup_meets = BeautifulSoup(out['response'].text, "html.parser")

    # time.sleep(0.5)
    for yr in YEARS:
        # yr = 2026
        print(yr)
        yr_sel = soup_meets.find("div", class_="smcfiltermenuyear dropdown-menu dropdown-menu-right")
        if not yr_sel:
            continue
        year_link = yr_sel.find("a", {"aria-label": f"Jahr: {yr} Monat: 1"})
        if not year_link:
            continue
        url_yr = urljoin(BASE_URL, year_link["href"])
        url_meetings = extract_meeting_urls(url_yr, BASE_URL)
        meetings +=  [{'gremium': gremium, 'year': yr} | get_url_info(x) | {'url': x} for x in url_meetings]
        time.sleep(0.5)

meeting=pd.DataFrame(meetings)

# SITZUNGSINFO: DATUM, ANWESENDE, TOPs, VORLAGE-ID, DATEIEN/ANLAGEN
for url in meeting.url:
    try:
        anw = extract_anwesende(url, BASE_URL)
        print(len(anw))
    except Exception as e:
        print(f"Anwesenheit fehlgeschlagen für {url}: {e}")
        anw = []

    out = fetch_page(url, cache_expiry=-1)
    sout = out['response']

    soup = BeautifulSoup(sout.text, "html.parser")

    sinf = extract_meeting_info(soup)
    sinf['source_url'] = url
    sinf['attendance'] = anw
    save_meeting(sinf)

    topps = extract_tops_and_files(soup, BASE_URL=BASE_URL, meeting_id=sinf['Sitzung'])
    for top in topps:
        save_top(top)

    # tdf = pd.DataFrame(topps)









