from Bio import Entrez
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 設置PubMed API的郵件地址及API key
def DownloadPubMedArticles(query, num_articles, api_key):
    Entrez.email = "azure0413@iir.csie.ncku.edu.tw"
    Entrez.api_key = "7183692faae8eed11d529217e61098834f08"

    xml_dir = "data"
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    # 搜索PubMed文章，获取文章ID列表
    handle = Entrez.esearch(db="pubmed", term=query, retmax=num_articles)
    record = Entrez.read(handle)
    handle.close()

    if not record["IdList"]:
        print("没有找到任何结果")
        return

    for pubmed_id in record["IdList"]:
        fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="xml", retmode="xml")
        
        # 读取字节串数据
        xml_data_bytes = fetch_handle.read()
        fetch_handle.close()

        # 将字节串解码为普通字符串
        xml_data_str = xml_data_bytes.decode('utf-8')

        # 生成文件名，并保存解码后的XML数据
        xml_file_name = os.path.join(xml_dir, f"X{pubmed_id}.xml")
        with open(xml_file_name, "w", encoding="utf-8") as xml_file:
            xml_file.write(xml_data_str)

        print(f"文章 {pubmed_id} 已成功下载并保存为 {xml_file_name}")

# 使用範例
api_key = "7183692faae8eed11d529217e61098834f08"  # 替換為你的PubMed API key
DownloadPubMedArticles("food safety", 30, api_key)