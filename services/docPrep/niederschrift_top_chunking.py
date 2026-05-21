import fitz  # PyMuPDF
import pdfplumber
import cv2
import numpy as np
from PIL import Image
import io, re, os, json
import pandas as pd

class Ratssiztung:

    def __init__(self, pdf_path, zoom = 2.0):
        self.pdf_path  = pdf_path
        self.zoom = zoom
        self.mat = fitz.Matrix(self.zoom, self.zoom)

    @staticmethod
    def p1_date(words):
        out = None
        pat = re.compile('[0-9]{2}\.[0-9]{2}\.[0-9]{4}')
        dpat = [x['text'] for x in words if re.match(pat, x['text'])]
        if len(dpat)==1:
            out = dpat[0]
        return out

    @staticmethod
    def get_top(words, n=1):
        pattern = re.compile(r"Tagesordnungspunkt", re.IGNORECASE)
        matches = []

        count = 0
        for i, word in enumerate(words):
            # print(word["text"])
            if pattern.match(word["text"]):
                # Collect all words that belong to the same line/block
                y0 = word["top"]
                y1 = word["bottom"]
                # block_words = [w for w in words if abs(w["top"] - y0) < 3]  # words on same horizontal line
                # x0 = min(w["x0"] for w in block_words)
                # x1 = max(w["x1"] for w in block_words)
                x0 = word['x0']
                x1 = word['x1']
                matches.append({
                    "text": word["text"],  # " ".join(w["text"] for w in block_words),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1
                })
                count += 1
                if count >= n: break

        return matches

    @staticmethod
    def get_top_rect(pix):
        #### get bounding box for each top using image rather than pdf
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")  # black/white conversion
        image_np = np.array(image)

        # === Horizontale Linien erkennen (OpenCV) ===
        _, binary = cv2.threshold(image_np, 200, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 500 and h < 10:
                line_boxes.append((x, y, w, h))

        line_ys = sorted([y for _, y, _, _ in line_boxes])

        # === duplicated Linien entfernen ===
        # deduped_lines = []
        # for y in line_ys:
        #     if not deduped_lines or abs(y - deduped_lines[-1]) > 10:
        #         deduped_lines.append(y)

        return line_ys

    def extract_top_outcomes(self, y_max, words):

        w_below = []
        for w in words:
            if (w['top'] * self.zoom > y_max):
                w_below.append(w['text'])

        return " ".join(w_below)

    def extract_top_info(self, top, y_min, y_max, words):

        out = {}

        if (top[0]['y0'] * self.zoom > y_min) and (top[0]['y1'] * self.zoom < y_max):

            top_cont = [w for w in words if (w['top'] * self.zoom > y_min) and (w['top'] * self.zoom < y_max)]

            line_top = None
            i_top_nr = None
            i_drucksache_nr = None
            content = []
            for i, c in enumerate(top_cont):
                if i == 0: line_top = c['top']
                if c['text'] == 'Tagesordnungspunkt':
                    i_top_nr = top_cont[i + 1]['text']
                    continue

                elif (c['text'] == 'Nr.') & (c['top'] == line_top):
                    i_drucksache_nr = top_cont[i + 1]['text']

                elif c['top'] > line_top:
                    content.append(c['text'])

            line_dist = np.unique([(x['bottom']) for x in top_cont])
            if (line_dist[-1] > 20):  # and (np.abs((line_dist[-1] * zoom) - rects[1]) < 10):
                pp_resp = [(x['text']) for x in top_cont if x['bottom'] == line_dist[-1]]
                person_resp = " ".join(pp_resp)
                content = content[0:-(len(pp_resp))]
            else:
                person_resp = None

            out = {'top_nr': i_top_nr, 'drucksache_nr': i_drucksache_nr, "person_resp": person_resp,
                   'topic': " ".join(content)}

        return out

    def get_pInfo(self, page_num, doc, words, pr=False):
        out = {}
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=self.mat)

        #### extract workds
        # with pdfplumber.open(pdf_path) as pdf:
        #     words = pdf.pages[page_num].extract_words(use_text_flow=True)
        # words = page.extract_words(use_text_flow=True)

        # plot overlay using matlab
        top = self.get_top(words, n=1)
        rects = self.get_top_rect(pix)

        if (len(rects) > 1) and top:
            y_min = rects[0]
            y_max = rects[1]
            out = self.extract_top_info(top, y_min, y_max, words)

            top_outcome = self.extract_top_outcomes(y_max, words)
            out['content1'] = top_outcome

            out['pars'] = {'page_start': page_num, 'rect_y_min': y_min, 'rect_y_max': y_max}
            # out['file'] = pdf_path.split('/')[-1]
            # print(pdf_path.split('/')[-1])
            if pr: print(out)
        # else:
        #     out['content'] = " ".join([w['text'] for w in words])

        return out

    @staticmethod
    def write_jsonl(file_path, record, type='a'):
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, type, encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")  # newline for NDJSON format

    @staticmethod
    def create_text_folder(pdf_path):
        path = os.path.join('corpus', 'text', os.path.basename(pdf_path).replace('.pdf', ''))
        os.makedirs(path or ".", exist_ok=True)
        return path

    def extract_tops(self):

        text_path = self.create_text_folder(self.pdf_path)

        doc = fitz.open(self.pdf_path)
        self.tops_extr = []
        top_content = []
        for page_num, page in enumerate(doc.pages()):
            with pdfplumber.open(self.pdf_path) as pdf:
                words = pdf.pages[page_num].extract_words(use_text_flow=True)

            if page_num == 0:
                # first page does not contain any tops
                dtt = self.p1_date(words)
            else:
                out = self.get_pInfo(page_num, doc, words, pr=False)

                if 'top_nr' in out:
                    # this means top was found
                    out['date'] = dtt
                    self.tops_extr.append(out)
                    ind_top = out['top_nr']

                    if top_content:
                        # print('appending top content to list[-2]')
                        self.tops_extr[-2]['content1'] = "".join(
                            [self.tops_extr[-2]['content1'], top_content])  # idx -2: not current but one before
                        self.tops_extr[-2]['pars']['end_page'] = (page_num - 1)
                        self.tops_extr[-2]['pars']['len_content1'] = len(self.tops_extr[-2]['content1'])
                        top_content = []
                    # else:
                    #     self.tops_extr[-2]['pars']['end_page'] = self.tops_extr[-2]['pars']['page_start']
                    #     self.tops_extr[-2]['pars']['len_content1'] = len(self.tops_extr[-2]['content1'])
                    # ### check for all words if location below rect_y_max, if so, add content
                elif self.tops_extr:  # top not found on page: assoc all text to previous top if it exists, else discard
                    print('collecting top content')
                    add = self.extract_top_outcomes(self.tops_extr[-1]['pars']['rect_y_min'], words)
                    # add = " ".join(words)
                    top_content = " ".join([top_content, add]) if top_content else add

        if top_content:
            self.tops_extr[-1]['content1'] = "".join(
                [self.tops_extr[-1]['content1'], top_content])  # idx -2: not current but one before
            self.tops_extr[-1]['pars']['end_page'] = (page_num)
            top_content = []

        fp = os.path.join(text_path, 'all_tops.jsonl')
        for i, top in enumerate(self.tops_extr):

            if i == 0:
                self.write_jsonl(fp, top, type='w')
            else:
                self.write_jsonl(fp, top, type='a')

            fp_single = os.path.join(text_path, f"top_{top['top_nr']}.jsonl")
            self.write_jsonl(fp_single, top, type='w')




# self = Ratssiztung(pdf_path='corpus/source_pdfs/20250224_duisburg_ratssitzung_öffentlich.pdf')
# self.extract_tops()
# df=pd.DataFrame([x for x in self.tops_extr if x])