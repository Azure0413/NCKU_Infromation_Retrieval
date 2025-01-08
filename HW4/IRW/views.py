import os
import re
from collections import Counter
from django.conf import settings
from django.shortcuts import render, redirect
from django.utils.html import mark_safe
from django.contrib import messages
import xml.etree.ElementTree as ET
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import Levenshtein
import numpy as np
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from django.core.files.storage import FileSystemStorage
from sentence_transformers import SentenceTransformer
from math import log, sqrt
import string

# Download NLTK data if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stop words
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

XML_FOLDER_PATH = os.path.join(settings.BASE_DIR, 'xml_folder')
CACHE_DIR = os.path.join(settings.BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)  # 確保目錄存在

def highlight_keyword(text, query):
    if query and text:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
        return mark_safe(highlighted)
    return text

def count_stop_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_word_count = sum(1 for word in words if word in stop_words)
    return stop_word_count

def count_sentences(text):
    abbreviations = r'\b(?:U\.S\.|e\.g\.|i\.e\.|Dr\.|Mr\.|Ms\.|Prof\.|Ltd\.|Inc\.|Jr\.|Sr\.)\b'
    abbreviation_pattern = re.compile(abbreviations, re.IGNORECASE)
    sentence_endings = re.finditer(r'[.!?]', text)

    if text == "":
        return 0

    sentence_count = 0

    for match in sentence_endings:
        end_pos = match.start()

        if end_pos + 1 < len(text) and text[end_pos + 1] == ' ' and end_pos + 2 < len(text) and (text[end_pos + 2].isupper() or text[end_pos + 2].isdigit()):
            before_punctuation = text[:end_pos].strip().split()[-1] if text[:end_pos].strip() else ''

            if not abbreviation_pattern.search(before_punctuation):
                sentence_count += 1

    sentence_count += 1
    return sentence_count

def count_letters_spaces(text):
    letters = sum(1 for char in text if char.isalpha())
    return letters

def count_stemmed_words(text):
    words = re.findall(r'\b\w+\b', text)
    stemmed_words = [ps.stem(word.lower()) for word in words]
    return len(set(stemmed_words))

def count_ascii_non_ascii(text):
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return ascii_count, non_ascii_count

def statistics(text):
    num_chars = count_letters_spaces(text)
    num_words = count_stemmed_words(text)
    num_sentences = count_sentences(text)
    num_ascii, num_non_ascii = count_ascii_non_ascii(text)

    return {
        'num_chars': num_chars,
        'num_words': num_words,
        'num_sentences': num_sentences,
        'num_ascii': num_ascii,
        'num_non_ascii': num_non_ascii,
    }

def extract_unique_words_from_titles():
    all_words = set()
    xml_folder = os.path.join(settings.BASE_DIR, 'xml_folder')

    for file in os.listdir( xml_folder):
        if file.endswith(".xml"):
            file_path = os.path.join(xml_folder, file)
            tree = ET.parse(file_path)
            root = tree.getroot()

            article_title_element = root.find('.//ArticleTitle')
            if article_title_element is not None:
                title_text = article_title_element.text
                if title_text:
                    words = word_tokenize(title_text)
                    all_words.update(word.lower() for word in words if word.isalpha())

    return all_words

def get_similar_words(query, all_words):
    distances = {word: Levenshtein.distance(query.lower(), word) for word in all_words}
    similar_words = sorted(distances, key=distances.get)[:10]
    return similar_words

MODEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'cbow_model.pkl')
MODEL_PATH = os.path.join(settings.BASE_DIR, 'cbow_model.pkl')

def preprocess_text(text):
    """將文字進行基本清理與處理。"""
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def calculate_similarity(query_words, sentence_words, model):
    """計算查詢字詞與句子的相似度。"""
    sentence_vec = np.mean([model.wv[word] for word in sentence_words if word in model.wv], axis=0)
    query_vec = np.mean([model.wv[word] for word in query_words if word in model.wv], axis=0)
    if sentence_vec is not None and query_vec is not None:
        similarity = np.dot(sentence_vec, query_vec) / (np.linalg.norm(sentence_vec) * np.linalg.norm(query_vec))
        return similarity if not np.isnan(similarity) else 0
    return 0

stop_words = set(stopwords.words("english"))

# tfidf_score
def calculate_method_1(abstract_text, query_words, model):
    sentences = sent_tokenize(abstract_text)
    if not sentences:
        return 0, []

    if isinstance(query_words, list):
        query = " ".join(query_words)
    else:
        query = query_words

    if not query.strip():
        print("Query words are empty!")
        return 0, []

    all_texts = sentences + [query]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)  # 稀疏矩陣
    query_vector = tfidf_matrix[-1]

    sentence_similarities = []
    for i, sentence in enumerate(sentences):
        sentence_vector = tfidf_matrix[i]  # 每個句子的向量
        similarity = cosine_similarity(sentence_vector, query_vector)[0][0]
        sentence_similarities.append((sentence, similarity))

    # 計算平均相似性
    total_similarity = np.mean([sim for _, sim in sentence_similarities]) if sentence_similarities else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

# cbow mean
def calculate_method_2(abstract_text, query_words, model):
    stop_words = set(stopwords.words('english'))

    sentences = sent_tokenize(abstract_text)
    sentence_similarities = []

    filtered_query_words = [qw for qw in query_words if qw.lower() not in stop_words]

    query_vectors = [model.wv[qw] for qw in filtered_query_words if qw in model.wv]
    if query_vectors:
        query_vector = np.mean(query_vectors, axis=0)  # 計算平均值
    else:
        query_vector = np.zeros(model.vector_size)  # 如果沒有詞彙對應於模型，使用零向量

    # 對於每個句子計算平均詞向量
    for sentence in sentences:
        sentence_words = sentence.split()
        filtered_sentence_words = [sw for sw in sentence_words if sw.lower() not in stop_words]  # 過濾停用詞
        sentence_vectors = [model.wv[sw] for sw in filtered_sentence_words if sw in model.wv]
        if sentence_vectors:
            sentence_vector = np.mean(sentence_vectors, axis=0)  # 計算平均值
        else:
            sentence_vector = np.zeros(model.vector_size)  # 如果句子中沒有有效的詞彙，使用零向量

        # 計算相似度
        if query_vector is not None and sentence_vector is not None:
            query_vector_2d = query_vector.reshape(1, -1)
            sentence_vector_2d = sentence_vector.reshape(1, -1)
            similarity = cosine_similarity(query_vector_2d, sentence_vector_2d)[0][0]
        else:
            similarity = 0  # 如果向量缺失，設置相似度為 0
        
        sentence_similarities.append((sentence, similarity))

    # 計算總相似度
    total_similarity = sum(sim for _, sim in sentence_similarities) / len(sentences) if sentences else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

# cbow concat
def calculate_method_3(abstract_text, query_words, model, max_length=50000):
    sentences = sent_tokenize(abstract_text)
    sentence_similarities = []

    def vector_concat(words, max_length):
        vectors = [model.wv[word] for word in words if word in model.wv]
        flat_vector = np.concatenate(vectors) if vectors else np.array([])
        if len(flat_vector) < max_length:
            return np.pad(flat_vector, (0, max_length - len(flat_vector)), 'constant')
        return flat_vector[:max_length]

    query_vector = vector_concat(query_words, max_length)

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_vector = vector_concat(sentence_words, max_length)
        similarity = (
            cosine_similarity([query_vector], [sentence_vector])[0][0]
            if len(query_vector) > 0 and len(sentence_vector) > 0
            else 0
        )
        sentence_similarities.append((sentence, similarity))

    total_similarity = sum(sim for _, sim in sentence_similarities) / len(sentences) if sentences else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

# tfidf concat cbow平均
def calculate_method_4(abstract_text, query_words, model_cbow):
    sentences = sent_tokenize(abstract_text)
    if not sentences:
        return 0, []

    if isinstance(query_words, list):
        query = " ".join(query_words)
    else:
        query = query_words

    if not query.strip():
        print("Query words are empty!")
        return 0, []

    # 計算TF-IDF權重
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences + [query])  # 包含句子和查詢的TF-IDF矩陣
    query_vector_tfidf = tfidf_matrix[-1]  # 查詢詞的TF-IDF向量
    
    # 使用CBOW模型為每個句子生成詞向量
    all_texts = sentences + [query]
    embeddings_cbow = []
    for text in all_texts:
        sentence_words = text.split()
        sentence_words = [word.strip(string.punctuation).lower() for word in sentence_words]
        word_vectors = []
        for word in sentence_words:
            if word in model_cbow.wv:  # Only use words in the model's vocabulary
                word_vectors.append(model_cbow.wv[word])
        # If no valid word vectors, use zero vector
        if word_vectors:
            embeddings_cbow.append(np.mean(word_vectors, axis=0))
        else:
            embeddings_cbow.append(np.zeros(model_cbow.vector_size))  # Zero vector if no words are in the vocab

    # Sentence embeddings (CBOW) and query embedding
    embeddings_cbow = np.array(embeddings_cbow)
    query_embedding_cbow = embeddings_cbow[-1]

    # 句子相似度計算
    sentence_similarities = []
    for i, sentence in enumerate(sentences):
        sentence_words = sentence.split()
        sentence_words = [word.strip(string.punctuation).lower() for word in sentence_words]
        sentence_vector_cbow = np.mean([model_cbow.wv[sw] for sw in sentence_words if sw in model_cbow.wv], axis=0) if sentence_words else np.zeros(model_cbow.vector_size)
        
        # 結合TF-IDF和CBOW
        sentence_vector_combined = np.concatenate((tfidf_matrix[i].toarray().flatten(), sentence_vector_cbow))

        # 計算查詢詞與句子的相似度
        query_vector_combined = np.concatenate((query_vector_tfidf.toarray().flatten(), query_embedding_cbow))

        similarity = cosine_similarity([query_vector_combined], [sentence_vector_combined])[0][0]
        sentence_similarities.append((sentence, similarity))

    # 計算平均相似性
    total_similarity = np.mean([sim for _, sim in sentence_similarities]) if sentence_similarities else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

# transformer
def calculate_method_5(abstract_text, query_words, model):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = sent_tokenize(abstract_text)
    if not sentences:
        return 0, []
    if isinstance(query_words, list):
        query = " ".join(query_words)
    else:
        query = query_words

    if not query.strip():
        print("Query words are empty!")
        return 0, []

    all_texts = sentences + [query]
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    query_embedding = embeddings[-1]

    sentence_similarities = []
    for i, sentence in enumerate(sentences):
        similarity = cosine_similarity([query_embedding], [embeddings[i]])[0][0]
        sentence_similarities.append((sentence, similarity))

    # 計算平均相似性
    total_similarity = np.mean([sim for _, sim in sentence_similarities]) if sentence_similarities else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

def calculate_method_6(abstract_text, query_words, model):
    sentences = sent_tokenize(abstract_text)
    if not sentences:
        return 0, []

    if isinstance(query_words, list):
        query = " ".join(query_words)
    else:
        query = query_words

    if not query.strip():
        print("Query words are empty!")
        return 0, []

    # Tokenize sentences and query
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    tokenized_query = word_tokenize(query.lower())

    # 初始化BM25模型
    bm25 = BM25Okapi(tokenized_sentences)

    # 計算每個句子對查詢的BM25分數
    bm25_scores = bm25.get_scores(tokenized_query)

    # 計算句子的嵌入向量（使用 CBOW 模型）
    sentence_vectors = []
    for sentence in tokenized_sentences:
        word_vectors = [
            model.wv[word.strip(string.punctuation)]
            for word in sentence if word.strip(string.punctuation) in model.wv
        ]
        if word_vectors:
            sentence_vectors.append(np.mean(word_vectors, axis=0))
        else:
            sentence_vectors.append(np.zeros(model.vector_size))  # 若句子無嵌入詞則用零向量代替

    # 計算查詢的嵌入向量
    query_vector = np.mean([
        model.wv[word.strip(string.punctuation)]
        for word in tokenized_query if word.strip(string.punctuation) in model.wv
    ], axis=0) if tokenized_query else np.zeros(model.vector_size)

    # 計算加權相似性
    sentence_similarities = []
    for i, (bm25_score, sentence_vector) in enumerate(zip(bm25_scores, sentence_vectors)):
        if np.linalg.norm(sentence_vector) > 0 and np.linalg.norm(query_vector) > 0:
            semantic_similarity = cosine_similarity([query_vector], [sentence_vector])[0][0]
            combined_similarity = bm25_score * semantic_similarity  # 使用 BM25 分數加權語意相似性
        else:
            combined_similarity = 0
        sentence_similarities.append((sentences[i], combined_similarity))

    # 提取所有相似性分數
    similarity_scores = [sim for _, sim in sentence_similarities]

    # 正規化相似性分數到 [0, 1]
    if similarity_scores:
        min_score = min(similarity_scores)
        max_score = max(similarity_scores)
        normalized_similarities = [
            ((sim - min_score) / (max_score - min_score)) if max_score > min_score else 0
            for sim in similarity_scores
        ]
        # 更新 sentence_similarities 為正規化後的值
        sentence_similarities = [
            (sentence, norm_sim) for (sentence, _), norm_sim in zip(sentence_similarities, normalized_similarities)
        ]

    # 計算平均相似性（使用正規化後的分數）
    total_similarity = np.mean([sim for _, sim in sentence_similarities]) if sentence_similarities else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

def calculate_method_7(abstract_text, query_words, model_cbow):
    sentences = sent_tokenize(abstract_text)
    if not sentences:
        return 0, []

    if isinstance(query_words, list):
        query = " ".join(query_words)
    else:
        query = query_words

    if not query.strip():
        print("Query words are empty!")
        return 0, []

    # 計算原始TF-IDF矩陣
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(sentences + [query])
    
    # 使用增強TF-IDF公式：Augmented TF
    term_frequencies = tfidf_matrix.toarray()
    max_term_frequencies = term_frequencies.max(axis=1, keepdims=True)
    augmented_tfidf = 0.5 + 0.5 * (term_frequencies / max_term_frequencies)
    
    # 查詢向量的增強TF-IDF
    query_vector_tfidf = augmented_tfidf[-1]  # 查詢詞的增強TF-IDF向量

    # 使用CBOW模型為每個句子生成詞向量
    all_texts = sentences + [query]
    embeddings_cbow = []
    for text in all_texts:
        sentence_words = text.split()
        sentence_words = [word.strip(string.punctuation).lower() for word in sentence_words]
        word_vectors = []
        for word in sentence_words:
            if word in model_cbow.wv:  # Only use words in the model's vocabulary
                word_vectors.append(model_cbow.wv[word])
        # If no valid word vectors, use zero vector
        if word_vectors:
            embeddings_cbow.append(np.mean(word_vectors, axis=0))
        else:
            embeddings_cbow.append(np.zeros(model_cbow.vector_size))  # Zero vector if no words are in the vocab

    # Sentence embeddings (CBOW) and query embedding
    embeddings_cbow = np.array(embeddings_cbow)
    query_embedding_cbow = embeddings_cbow[-1]

    # 句子相似度計算
    sentence_similarities = []
    for i, sentence in enumerate(sentences):
        sentence_words = sentence.split()
        sentence_words = [word.strip(string.punctuation).lower() for word in sentence_words]
        sentence_vector_cbow = np.mean([model_cbow.wv[sw] for sw in sentence_words if sw in model_cbow.wv], axis=0) if sentence_words else np.zeros(model_cbow.vector_size)
        
        # 結合增強TF-IDF和CBOW
        sentence_vector_combined = np.concatenate((augmented_tfidf[i], sentence_vector_cbow))

        # 查詢向量的增強TF-IDF與CBOW結合
        query_vector_combined = np.concatenate((query_vector_tfidf, query_embedding_cbow))

        # 計算查詢詞與句子的相似度
        similarity = cosine_similarity([query_vector_combined], [sentence_vector_combined])[0][0]
        sentence_similarities.append((sentence, similarity))

    # 計算平均相似性
    total_similarity = np.mean([sim for _, sim in sentence_similarities]) if sentence_similarities else 0
    return total_similarity, sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:5]

def index_view(request):
    query = request.GET.get('q')
    selected_year = request.GET.get('year')
    method = request.GET.get('method', '1')
    files = []
    years = set()
    categories = ['covid-19', 'enterovirus']
    result_limit = 20

    # 檢查或生成 CBOW 模型
    if not os.path.exists(MODEL_PATH):
        abstracts = []
        for category in categories:
            category_folder = os.path.join(settings.BASE_DIR, 'xml_folder', category)
            if os.path.exists(category_folder):
                for file in os.listdir(category_folder):
                    if file.endswith(".xml"):
                        file_path = os.path.join(category_folder, file)
                        tree = ET.parse(file_path)
                        root = tree.getroot()
                        abstract_elements = root.findall('.//AbstractText')
                        abstract_text = ' '.join([ET.tostring(el, encoding='unicode') for el in abstract_elements])
                        abstract_text = re.sub(r'<[^>]+>', '', abstract_text)
                        abstracts.append(preprocess_text(abstract_text))

        model = Word2Vec(sentences=abstracts, vector_size=1000, window=5, min_count=1, sg=0)
        model.save(MODEL_PATH)
    else:
        model = Word2Vec.load(MODEL_PATH)

    query_words = preprocess_text(query) if query else []

    for category in categories:
        category_folder = os.path.join(settings.BASE_DIR, 'xml_folder', category)
        if not os.path.exists(category_folder):
            continue

        for file in os.listdir(category_folder):
            if file.endswith(".xml"):
                file_path = os.path.join(category_folder, file)
                tree = ET.parse(file_path)
                root = tree.getroot()

                article_title_element = root.find('.//ArticleTitle')
                year_element = root.find('.//PubDate/Year')
                abstract_elements = root.findall('.//AbstractText')
                doi_element = root.find('.//ArticleId[@IdType="doi"]')
                pmid_element = root.find('.//ArticleId[@IdType="pubmed"]')

                article_title = article_title_element.text if article_title_element is not None else 'N/A'
                pub_year = year_element.text if year_element is not None else 'N/A'
                doi = doi_element.text if doi_element is not None else 'N/A'
                pmid = pmid_element.text if pmid_element is not None else 'N/A'

                abstract_text = ''.join([ET.tostring(el, encoding='unicode') for el in abstract_elements])
                abstract_text = re.sub(r'<[^>]+>', '', abstract_text)

                if pub_year != 'N/A':
                    years.add(pub_year)

                if selected_year and selected_year != pub_year:
                    continue

                # 計算統計數據
                num_chars = len(abstract_text)
                num_words = len(abstract_text.split())
                num_sentences = len(sent_tokenize(abstract_text))
                num_ascii = sum(1 for c in abstract_text if ord(c) < 128)
                num_non_ascii = num_chars - num_ascii

                # 根據選擇的相似度算法計算
                if method == '1':
                    total_similarity, top_sentences = calculate_method_1(abstract_text, query_words, model)
                elif method == '2':
                    total_similarity, top_sentences = calculate_method_2(abstract_text, query_words, model)
                elif method == '3':
                    total_similarity, top_sentences = calculate_method_3(abstract_text, query_words, model)
                elif method == '4':
                    total_similarity, top_sentences = calculate_method_4(abstract_text, query_words, model)
                elif method == '5':
                    total_similarity, top_sentences = calculate_method_5(abstract_text, query_words, model)
                elif method == '6':
                    total_similarity, top_sentences = calculate_method_6(abstract_text, query_words, model)
                elif method == '7':
                    total_similarity, top_sentences = calculate_method_7(abstract_text, query_words, model)
                else:
                    raise ValueError("Invalid method selected")
                
                # 將 top_sentences 中的句子加上 HTML 標註，並將其插入到原始 abstract 中
                highlighted_abstract = abstract_text  # 開始時是原始的摘要
                for sentence, _ in top_sentences:
                    highlighted_abstract = highlighted_abstract.replace(sentence, f"<span class='highlight'>{sentence}</span>")

                # 在 file 中設置處理過的摘要
                files.append({
                    'filename': file,
                    'category': category,
                    'title': article_title,
                    'pub_year': pub_year,
                    'abstract': highlighted_abstract,  # 使用帶有顏色標註的摘要
                    'similarity': total_similarity,
                    'top_sentences': [{'text': sent, 'similarity': sim} for sent, sim in top_sentences],
                    'doi': doi,
                    'pmid': pmid,
                    'statistics': {
                        'num_chars': num_chars,
                        'num_words': num_words,
                        'num_sentences': num_sentences,
                        'num_ascii': num_ascii,
                        'num_non_ascii': num_non_ascii,
                    },
                })
            if len(files) >= result_limit:
                break

    # 根據 similarity 排序
    files = sorted(files, key=lambda x: x['similarity'], reverse=True)[:result_limit]
    result_message = f"顯示前 {result_limit} 筆相關結果，若需要完整資料，請精確化搜尋條件。"

    return render(request, 'index.html', {
        'files': files,
        'query': query,
        'years': sorted(years),
        'result_message': result_message,
    })

def file_analysis(request, filename):
    # 定義資料夾名稱作為類別標籤
    categories = ['covid-19', 'enterovirus']
    file_path = None

    # 檢查每個子資料夾，找出正確的檔案路徑
    for category in categories:
        folder_path = os.path.join(XML_FOLDER_PATH, category)
        potential_file_path = os.path.join(folder_path, filename)
        
        # 若檔案存在於該路徑中，則設置 file_path 為該路徑
        if os.path.exists(potential_file_path):
            file_path = potential_file_path
            break

    # 若 file_path 仍為 None，表示檔案不存在於任何子資料夾中
    if file_path is None:
        messages.error(request, f"檔案 '{filename}' 不存在於 xml_folder 的任何子資料夾中！")
        return redirect('index')  # 跳轉回主頁面

    # 確保檔案存在後再繼續解析處理
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        messages.error(request, f"無法解析檔案 '{filename}'，檔案可能已損壞或格式不正確。")
        return redirect('index')

    article_title_element = root.find('.//ArticleTitle')
    abstract_text_elements = root.findall('.//AbstractText')

    article_title = article_title_element.text if article_title_element is not None else 'N/A'
    abstract_text = ''.join([ET.tostring(el, encoding='unicode') for el in abstract_text_elements])

    abstract_text = re.sub(r'\s{2,}', ' ', abstract_text).strip()
    abstract_text = re.sub(r'<[^>]+>', '', abstract_text)
    char_count = len(abstract_text)
    word_count = len(abstract_text.split())
    sentence_count = count_sentences(mark_safe(abstract_text))

    # Advanced statistics
    letter_count = count_letters_spaces(abstract_text)
    stemmed_word_count = count_stemmed_words(abstract_text)
    stop_word_count = count_stop_words(abstract_text)
    ascii_count, non_ascii_count = count_ascii_non_ascii(abstract_text)

    query = request.GET.get('q', '').strip()

    word_tokens = word_tokenize(abstract_text)
    filtered_stopwords = [word for word in word_tokens if word.lower() in stop_words]

    keyword_count = 0
    if query:
        abstract_text = highlight_keyword(abstract_text, query)
        keyword_count = len(re.findall(re.escape(query), abstract_text, re.IGNORECASE))

    context = {
        'filename': filename,
        'article_title': mark_safe(article_title),
        'abstract_text': mark_safe(abstract_text), 
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'letter_count': letter_count,
        'space_count': char_count - letter_count,
        'stemmed_word_count': stemmed_word_count,
        'stop_word_count': stop_word_count,
        'query': query,
        'stopwords': filtered_stopwords,
        'ascii_count': ascii_count, 
        'non_ascii_count': non_ascii_count, 
        'keyword_count': keyword_count
    }

    return render(request, 'file_analysis.html', context)

# Check if chart exists
def check_chart_exists(chart_filename):
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    return os.path.exists(chart_path)

# Get word counts from folder
def get_word_counts_from_folder(folder_path):
    word_counts = Counter()
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Extract abstract text
                abstract_text_elements = root.findall('.//AbstractText')
                abstract_text = ''.join([ET.tostring(el, encoding='unicode') for el in abstract_text_elements])
                abstract_text = re.sub(r'<[^>]+>', '', abstract_text)

                if abstract_text:
                    words = abstract_text.lower().split()
                    word_counts.update(words)
            except ET.ParseError as e:
                print(f"Error parsing {file_path}: {e}")
    return word_counts

# Apply Porter Stemmer
def apply_stemming(word_counts):
    ps = PorterStemmer()
    stemmed_counts = Counter()
    for word, count in word_counts.items():
        stemmed_word = ps.stem(word)
        stemmed_counts[stemmed_word] += count
    return stemmed_counts

# Remove stop words
def remove_stopwords(word_counts):
    stop_words = set(stopwords.words('english'))
    filtered_counts = Counter({word: count for word, count in word_counts.items() if word not in stop_words})
    return filtered_counts 

def generate_word_cloud(word_counts, chart_filename):
    if not word_counts:
        print("word_counts is empty. No data to generate word cloud.")
        return

    font_path = 'C:/Windows/Fonts/Arial.ttf' 
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def distribution_view(request):
    chart_paths = {}

    # Define folder paths
    folder_paths = {
        'covid-19': os.path.join(settings.MEDIA_ROOT, 'xml_folder', 'covid-19'),
        'enterovirus': os.path.join(settings.MEDIA_ROOT, 'xml_folder', 'enterovirus')
    }

    combined_word_counts = {}
    combined_filtered_counts = {}

    for folder_name, folder_path in folder_paths.items():
        word_counts = get_word_counts_from_folder(folder_path)
        filtered_counts = remove_stopwords(word_counts)

        # Update combined data
        for word, count in word_counts.items():
            combined_word_counts[word] = combined_word_counts.get(word, 0) + count
        for word, count in filtered_counts.items():
            combined_filtered_counts[word] = combined_filtered_counts.get(word, 0) + count

        # Generate word cloud for this folder
        chart_filename_wordcloud = f'{folder_name}_wordcloud.png'
        if not check_chart_exists(chart_filename_wordcloud):
            generate_word_cloud(word_counts, chart_filename_wordcloud)
        chart_paths[f'{folder_name} Word Cloud'] = f'static/charts/{chart_filename_wordcloud}'

        # Generate filtered word cloud for this folder
        chart_filename_wordcloud_filtered = f'{folder_name}_wordcloud_stopword.png'
        if not check_chart_exists(chart_filename_wordcloud_filtered):
            generate_word_cloud(filtered_counts, chart_filename_wordcloud_filtered)
        chart_paths[f'{folder_name} Word Cloud (Stopwords Removed)'] = f'static/charts/{chart_filename_wordcloud_filtered}'

    # Generate combined word clouds
    combined_chart_filename = 'combined_wordcloud.png'
    if not check_chart_exists(combined_chart_filename):
        generate_word_cloud(combined_word_counts, combined_chart_filename)
    chart_paths['Combined Word Cloud'] = f'static/charts/{combined_chart_filename}'

    combined_filtered_chart_filename = 'combined_wordcloud_stopword.png'
    if not check_chart_exists(combined_filtered_chart_filename):
        generate_word_cloud(combined_filtered_counts, combined_filtered_chart_filename)
    chart_paths['Combined Word Cloud (Stopwords Removed)'] = f'static/charts/{combined_filtered_chart_filename}'

    return render(request, 'distribution.html', {'chart_paths': chart_paths})

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    return [word for word in words if word.isalpha() and word not in stop_words]

def extract_abstract_texts(dataset_folder):
    all_abstracts = []

    for file in os.listdir(dataset_folder):
        if file.endswith(".xml"):
            file_path = os.path.join(dataset_folder, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                abstract_text_elements = root.findall('.//AbstractText')

                # 將多個 <AbstractText> 合併
                abstract_text = ''.join(
                    [ET.tostring(el, encoding='unicode') for el in abstract_text_elements]
                )
                abstract_text = re.sub(r'<[^>]+>', '', abstract_text)  # 移除 XML 標籤
                processed_words = preprocess_text(abstract_text)
                if processed_words:  # 避免空白內容
                    all_abstracts.append(processed_words)
            except ET.ParseError:
                print(f"Error parsing file: {file_path}")  # 日誌記錄
                continue

    return all_abstracts

def extract_abstract_text(dataset_folder):
    all_abstracts = []

    # 如果是單一檔案而不是資料夾
    if os.path.isfile(dataset_folder):
        file_path = dataset_folder
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            abstract_text_elements = root.findall('.//AbstractText')

            # 將多個 <AbstractText> 合併
            abstract_text = ''.join(
                [ET.tostring(el, encoding='unicode') for el in abstract_text_elements]
            )
            abstract_text = re.sub(r'<[^>]+>', '', abstract_text)  # 移除 XML 標籤
            processed_words = preprocess_text(abstract_text)
            if processed_words:  # 避免空白內容
                all_abstracts.append(processed_words)
        except ET.ParseError:
            print(f"Error parsing file: {file_path}")  # 日誌記錄
    else:
        # 如果是資料夾，遍歷該資料夾中的檔案
        for file in os.listdir(dataset_folder):
            if file.endswith(".xml"):
                file_path = os.path.join(dataset_folder, file)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    abstract_text_elements = root.findall('.//AbstractText')

                    # 將多個 <AbstractText> 合併
                    abstract_text = ''.join(
                        [ET.tostring(el, encoding='unicode') for el in abstract_text_elements]
                    )
                    abstract_text = re.sub(r'<[^>]+>', '', abstract_text)  # 移除 XML 標籤
                    processed_words = preprocess_text(abstract_text)
                    if processed_words:  # 避免空白內容
                        all_abstracts.append(processed_words)
                except ET.ParseError:
                    print(f"Error parsing file: {file_path}")  # 日誌記錄
                    continue

    return all_abstracts

def generate_bar_chart(word_scores, chart_filename, title, color):
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    if os.path.exists(chart_path):
        return chart_path

    if not word_scores:
        return None

    words, scores = zip(*word_scores[:30])
    plt.figure(figsize=(12, 6))
    plt.bar(words, scores, color=color)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Words")
    plt.ylabel("Scores")
    plt.tight_layout()

    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def compute_tfidf_sklearn(documents):
    corpus = [' '.join(doc) for doc in documents]
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.sum(axis=0).A1))
        return tfidf_scores
    except Exception as e:
        print(f"Error in compute_tfidf_sklearn: {e}")
        return {}

def compute_bm25_rankbm25(documents):
    bm25_model = BM25Okapi(documents)
    return bm25_model

def compute_augmented_tfidf(documents, idf):
    augmented_tfidf_scores = defaultdict(float)
    for doc in documents:
        max_freq = max(Counter(doc).values())
        tf = {word: 0.5 + 0.5 * (freq / max_freq) for word, freq in Counter(doc).items()}
        for word, tf_score in tf.items():
            augmented_tfidf_scores[word] += tf_score * idf[word]
    return augmented_tfidf_scores

def compute_idf(documents):
    N = len(documents)
    idf = {}
    all_words = set(word for doc in documents for word in doc)
    for word in all_words:
        containing_docs = sum(1 for doc in documents if word in doc)
        idf[word] = math.log((N + 1) / (containing_docs + 1)) + 1
    return idf

def calculate_user_tf(tf_formula, word_count, total_words):
    if tf_formula == "log(tf)":
        return log(word_count + 1)
    elif tf_formula == "sqrt(tf)":
        return sqrt(word_count)
    else:  # Default to plain TF
        return word_count / total_words

def calculate_user_idf(idf_formula, total_documents, word_document_count):
    if idf_formula == "log(N / df)":
        return log(total_documents / (word_document_count + 1))
    elif idf_formula == "log(1 + (N / df))":
        return log(1 + total_documents / (word_document_count + 1))
    else:  # Default to log(N - df)
        # Prevent log of zero or negative values
        if total_documents > word_document_count:
            return log(total_documents - word_document_count)
        else:
            return 0

def calculate_normal_tf(word_count, total_words):
    return word_count / total_words  # 正常的 TF 計算方式

def calculate_normal_idf(total_documents, word_document_count):
    return log(total_documents / (word_document_count + 1))  # 正常的 IDF 計算方式

def w2v_view(request):
    xml_folder = os.path.join(settings.BASE_DIR, 'xml_folder')
    datasets = ["covid-19", "enterovirus"]
    results = {}
    dataset_documents = {}
    uploaded_tables = {}
    analysis_results = {"abstract_scores": []}

    for dataset in datasets:
        dataset_folder = os.path.join(xml_folder, dataset)
        dataset_documents[dataset] = extract_abstract_texts(dataset_folder)

    combined_documents = dataset_documents.get("covid-19", []) + dataset_documents.get("enterovirus", [])
    dataset_documents["combined"] = combined_documents
    uploaded_abstract = ""

    for dataset, documents in dataset_documents.items():
        if not documents:
            continue

        idf = compute_idf(documents)
        avg_doc_len = sum(len(doc) for doc in documents) / len(documents)

        tfidf_scores = compute_tfidf_sklearn(documents)
        bm25_model = compute_bm25_rankbm25(documents)
        
        bm25_scores = {}
        for word in idf:
            word_scores = bm25_model.get_scores([word])
            bm25_scores[word] = sum(word_scores)

        augmented_tfidf_scores = compute_augmented_tfidf(documents, idf)

        top_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        top_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        top_augmented_tfidf = sorted(augmented_tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:30]

        tfidf_chart_file = f"{dataset}_tfidf.png"
        bm25_chart_file = f"{dataset}_bm25.png"
        augmented_chart_file = f"{dataset}_augmented_tfidf.png"

        tfidf_chart = generate_bar_chart(top_tfidf, tfidf_chart_file, f"TF-IDF: {dataset}", 'blue')
        bm25_chart = generate_bar_chart(top_bm25, bm25_chart_file, f"BM25: {dataset}", 'green')
        augmented_chart = generate_bar_chart(top_augmented_tfidf, augmented_chart_file, f"Augmented TF-IDF: {dataset}", 'red')

        results[dataset] = {
            "tfidf_chart": tfidf_chart,
            "bm25_chart": bm25_chart,
            "augmented_chart": augmented_chart,
        }

    if request.method == "POST" and request.FILES.get("xml_file"):
        xml_file = request.FILES["xml_file"]
        fs = FileSystemStorage()

        temp_dir = os.path.join(settings.BASE_DIR, 'static', 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        uploaded_file = request.FILES['xml_file']
        tree = ET.parse(uploaded_file)
        root = tree.getroot()

        abstract = root.find('.//AbstractText')

        if abstract is not None and abstract.text is not None:
            uploaded_abstract = abstract.text.strip()

        temp_path = fs.save(os.path.join('static', 'temp', "temp.xml"), xml_file)
        temp_file_path = fs.path(temp_path)

        uploaded_docs = extract_abstract_text(temp_file_path)
        total_documents = len(uploaded_docs)

        # 從表單中獲取使用者選擇的公式
        tf_formula_user = request.POST.get("tf_formula", "log(tf)")  # 默認為 "log(tf)"
        idf_formula_user = request.POST.get("idf_formula", "log(N / df)")  # 默認為 "log(N / df)"

        idf_normal = {}
        tfidf_scores_normal = {}
        tfidf_scores_user = {}

        # 計算正常的 TF 和 IDF
        for doc in uploaded_docs:
            word_count = {}
            total_words = 0

            if isinstance(doc, list):
                doc = " ".join(doc)

            for word in doc.split():
                word_count[word] = word_count.get(word, 0) + 1
                total_words += 1

            for word, count in word_count.items():
                tf_value = calculate_normal_tf(count, total_words)
                tfidf_scores_normal[word] = tfidf_scores_normal.get(word, 0) + tf_value

                word_document_count = sum(1 for d in uploaded_docs if word in d)
                idf_normal[word] = calculate_normal_idf(total_documents, word_document_count)

        # 計算使用者自訂的 TF 和 IDF
        for doc in uploaded_docs:
            word_count = {}
            total_words = 0

            if isinstance(doc, list):
                doc = " ".join(doc)

            for word in doc.split():
                word_count[word] = word_count.get(word, 0) + 1
                total_words += 1

            for word, count in word_count.items():
                # 使用者定義的 TF 和 IDF
                tf_value_user = calculate_user_tf(tf_formula_user, count, total_words)  # 使用選擇的公式
                tfidf_scores_user[word] = tfidf_scores_user.get(word, 0) + tf_value_user

                word_document_count_user = sum(1 for d in uploaded_docs if word in d)
                idf_user = calculate_user_idf(idf_formula_user, total_documents, word_document_count_user)

        # 計算 TF-IDF
        tf_idf_table_normal = []
        tf_idf_table_user = []

        for word in idf_normal:
            tf_idf_table_normal.append({
                "word": word,
                "tf": calculate_normal_tf(tfidf_scores_normal.get(word, 0), total_words),
                "idf": idf_normal.get(word, 0),
                "tf_idf": calculate_normal_tf(tfidf_scores_normal.get(word, 0), total_words) * idf_normal.get(word, 0)
            })

        for word in tfidf_scores_user:
            tf_idf_table_user.append({
                "word": word,
                "tf": calculate_user_tf(tf_formula_user, tfidf_scores_user.get(word, 0), total_words),
                "idf": calculate_user_idf(idf_formula_user, total_documents, word_document_count_user),
                "tf_idf": calculate_user_tf(tf_formula_user, tfidf_scores_user.get(word, 0), total_words) * calculate_user_idf(idf_formula_user, total_documents, word_document_count_user)
            })

        uploaded_tables = {
            "tf_idf_table_normal": tf_idf_table_normal[:10],  # 顯示前 10 個
            "tf_idf_table_user": tf_idf_table_user[:10]  # 顯示前 10 個
        }

    return render(request, 'w2v.html', {
        'results': results,
        'uploaded_tables': uploaded_tables,
        'analysis_results': analysis_results,
        'uploaded_abstract': uploaded_abstract
    })