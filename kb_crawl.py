from crawler import crawl
import os
from pathlib import Path





kb_urls = [
"https://www.iseas.edu.sg/",
"https://www.rsis.edu.sg/research/idss/",
"https://www.siiaonline.org/",
"https://research.nus.edu.sg/eai/",
"https://www.prsgroup.com/",
"https://lkyspp.nus.edu.sg/cag",
"https://lkyspp.nus.edu.sg/aci",
"https://lkyspp.nus.edu.sg/ips",
"https://esi.nus.edu.sg/",
"https://mei.nus.edu.sg/",
"https://cser.energy/",
"https://www.rsis.edu.sg/",
"https://cil.nus.edu.sg/",
"https://www.clc.gov.sg/",
"https://lkyspp.nus.edu.sg/ies"

]

for start_url in kb_urls:
    print(f"{start_url=}")
    dir_name = start_url.replace('/', "-").replace('.', '-').replace(":", "")
    dest_data_path = Path("/mnt/ecf82360-d01d-4966-b234-c47ea01078db/datas1220/")
    save_dir = dest_data_path / dir_name
    os.makedirs(save_dir, exist_ok=True)
    crawled_pages = crawl(start_url, save_dir, max_pages=1000_000)
    print(f"{len(crawled_pages)=}")




