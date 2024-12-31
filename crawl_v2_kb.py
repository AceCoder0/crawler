import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlsplit
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

def download_pdf(link, save_path='./'):
    # 检查提供的URL是否为空或无效
    # time.sleep(1)
    if not link:
        print("无效的URL")
        return
    
    # 获取文件名
    url_path = urlsplit(link).path
    file_name = os.path.basename(url_path)
    
    # 检查URL是否指向PDF文件
    if not file_name.lower().endswith('.pdf'):
        print(f"提供的链接 {link} 不像是指向一个PDF文件")
        return
    
    try:
        # 发送HTTP请求获取PDF文件
        response = requests.get(link, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        # 定义本地文件路径
        local_file_path = save_path / file_name
        
        # 下载并保存PDF文件
        with open(local_file_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)
        
        print(f"PDF已成功下载并保存到: {local_file_path}")
    except requests.RequestException as e:
        print(f"下载失败: {e}")
    return local_file_path

visited_pages = set()

# 爬取网站
def crawl(url, save_dir, max_pages=100, max_retry=3):
    pages_to_visit = deque([url])
    global visited_pages
    html_dir = save_dir / "raw_htmls"
    pdf_dir = save_dir / "pdfs"
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    table = []
    tb_fp = save_dir / "table.jsonl"
    # level=0
    while pages_to_visit and len(visited_pages) < max_pages:
        current_page = pages_to_visit.popleft()
        if (current_page in visited_pages) or ('.jpg' in current_page):
            continue
        print(f"Crawling: {current_page}")
        for i in range(max_retry):
            try:
                if '.pdf' in current_page:
                    download_pdf(current_page, pdf_dir)
                    visited_pages.add(current_page)
                    break
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
    # start_url = 'https://www.zaobao.com/'  # 目标网站URL
    dir_name = start_url.replace('/', "-").replace('.', '-').replace(":", "")
    dest_data_path = Path("/mnt/ecf82360-d01d-4966-b234-c47ea01078db/datas1224/kb/")
    save_dir = dest_data_path / dir_name
    os.makedirs(save_dir, exist_ok=True)
    crawled_pages = crawl(start_url, save_dir, max_pages=1000_000)
    print(f"{len(crawled_pages)=}")