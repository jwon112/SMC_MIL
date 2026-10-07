"""Acquire the public source copies selected in the 2026-09-30 review.

This downloads source PDFs only. Registration requires separate human review.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import ssl
import certifi

from fetch_pmc_pdf import resolve_pdf_key, BUCKET

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "li2021_dsmil": "https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf",
    "chen2022_hipt": "https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.pdf",
    "zhang2022_dtfd": "https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.pdf",
    "guo2017_calibration": "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf",
    "jaume2024_madeleine": "https://arxiv.org/pdf/2408.02859v1",
    "vorontsov2024_virchow": "PMC11485232",
    "dwivedi2022_multistain_graph": "https://openaccess.thecvf.com/content/CVPR2022W/CVMI/papers/Dwivedi_Multi_Stain_Graph_Fusion_for_Multimodal_Integration_in_Pathology_CVPRW_2022_paper.pdf",
    "peyster2024_clinical_trajectory": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10940208/",
    "howard2021_site_signatures": "PMID34285218",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "SMC-literature-review/1.0"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
        return response.read()


def acquire(item):
    paper_id, source = item
    try:
        if source.startswith("PMID"):
            payload = json.loads(fetch("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids=" + source[4:] + "&format=json"))
            source = payload["records"][0]["pmcid"]
        if source.startswith("PMC"):
            source = BUCKET + "/" + resolve_pdf_key(source)
        path = ROOT / "paper_origin" / (paper_id + ".pdf")
        if path.exists():
            data = path.read_bytes()
        else:
            data = fetch(source)
            if not data.startswith(b"%PDF-"):
                raise ValueError("not a PDF")
            path.write_bytes(data)
        result = {"paper_id": paper_id, "source_url": source, "filename": path.name,
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "accessed_on": "2026-09-30"}
        print("OK", paper_id, len(data), flush=True)
        return result
    except Exception as exc:
        print("ERROR", paper_id, str(exc), flush=True)
        return {"paper_id": paper_id, "error": str(exc)}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(acquire, SOURCES.items()))
    path = ROOT / "derived" / "acquisition_20260930.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(any("error" in result for result in results))
