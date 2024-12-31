import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import os
from pathlib import Path
import hashlib
import string
import json

def is_valid(url):
    # 过滤掉非HTTP/HTTPS链接和其他无效链接
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def url_to_filename(url):
    # 使用 SHA-256 算法计算哈希值，并转换为十六进制字符串
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig

def sanitize_filename(filename):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    sanitized = ''.join(c if c in valid_chars else '_' for c in filename)
    return sanitized[:255]  # 限制文件名长度

# 结合哈希与清理
def url_to_safe_filename(url):
    hash_value = url_to_filename(url)
    return sanitize_filename(hash_value)
def get_all_website_links(url):
    urls = set()
    domain_name = urlparse(url).netloc
    page = requests.get(url).content
    soup = BeautifulSoup(page, "html.parser")

    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            continue
        if href in urls:
            continue
        if domain_name not in href:
            continue
        urls.add(href)
    return urls, str(page)

visited_pages = set()

# 爬取网站
def crawl(url, save_dir, max_pages=100, max_retry=3):
    pages_to_visit = deque([url])
    global visited_pages
    html_dir = save_dir / "raw_htmls"
    os.makedirs(html_dir, exist_ok=True)
    table = []
    tb_fp = save_dir / "table.jsonl"
    # level=0
    while pages_to_visit and len(visited_pages) < max_pages:
        current_page = pages_to_visit.popleft()
        if current_page in visited_pages:
            continue
        print(f"Crawling: {current_page}")
        for i in range(max_retry):
            try:
                links, html = get_all_website_links(current_page)
                for link in links:
                    # if (not (link in visited_pages)) and (not (link not in pages_to_visit)):
                    pages_to_visit.append(link)
                visited_pages.add(current_page)
                htmlname = url_to_safe_filename(current_page)+'.html'
                html_path = html_dir / htmlname
                with open(html_path, 'w') as f:
                    f.write(html)
                table.append({
                    'status': 'GET',
                    'url': current_page,
                    'html_path': html_path.as_posix(),
                    'exception': None
                })
                with open(tb_fp, 'a', encoding='utf-8') as f:
                    json_str = json.dumps({
                    'status': 'GET',
                    'url': current_page,
                    'html_path': html_path.as_posix(),
                    'exception': None
                    }, ensure_ascii=False)
                    f.write(json_str+'\n')
                break
            except Exception as e:
                print(f"Failed to crawl {current_page}: {e}")
                if i==max_retry-1:
                    table.append({
                        "status": 'FAIL',
                        "url": current_page,
                        "html_path": None,
                        "exception": str(e)
                    })
                    with open(tb_fp, 'a', encoding='utf-8') as f:
                        json_str = json.dumps({
                        "status": 'FAIL',
                        "url": current_page,
                        "html_path": None,
                        "exception": str(e)
                        }, ensure_ascii=False)
                        f.write(json_str+'\n')
    with open(save_dir / "table.json", 'w') as jsonf:
        json.dump(table, jsonf, ensure_ascii=False, indent=4)
    return visited_pages


# 调用爬虫函数
start_url = 'https://www.zaobao.com/'  # 目标网站URL
dir_name = start_url.replace('/', "-").replace('.', '-').replace(":", "")
dest_data_path = Path("/mnt/ecf82360-d01d-4966-b234-c47ea01078db/datas1224/")
save_dir = dest_data_path / dir_name
os.makedirs(save_dir, exist_ok=True)
crawled_pages = crawl(start_url, save_dir, max_pages=1000_000)
print(f"{len(crawled_pages)=}")