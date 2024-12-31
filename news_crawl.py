from crawler import crawl
import os
from pathlib import Path


news_urls = [
    # "https://www.zaobao.com.sg/",
"https://www.bharian.com.my/",
"https://www.shinmin.sg/",
"https://www.thumbsuptherapy.sg/",
"https://www.herworld.com/",
"https://www.straitstimes.com/",
"https://www.businesstimes.com.sg/",
"https://tnp.straitstimes.com/",
"https://www.todayonline.com/",
"https://www.nuyou.com.sg/",
"https://www.tamilmurasu.com.sg/",
"https://www.mediacorp.sg/",
"https://www.asiaone.com/",
"https://www.harpersbazaar.com.sg/",
"https://www.singaporestar.com/",
"http://news.nanyangpost.com/",
"https://www.channelnewsasia.com/us",
"https://www.yan.sg/",
"https://mothership.sg/",
"https://www.thepeakmagazine.com.sg/",
"https://www.techinasia.com/",
"https://www.homeanddecor.com.sg/",
"https://www.tabla.com.sg/",
"https://stomp.straitstimes.com/",
"https://www.femalemag.com.sg/",
]

for start_url in news_urls:
    print(f"{start_url=}")
    dir_name = start_url.replace('/', "-").replace('.', '-').replace(":", "")
    dest_data_path = Path("/mnt/ecf82360-d01d-4966-b234-c47ea01078db/datas1220/")
    save_dir = dest_data_path / dir_name
    os.makedirs(save_dir, exist_ok=True)
    crawled_pages = crawl(start_url, save_dir, max_pages=1000_000)
    print(f"{len(crawled_pages)=}")
