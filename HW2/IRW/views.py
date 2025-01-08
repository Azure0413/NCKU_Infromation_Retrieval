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
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import Levenshtein
import numpy as np

# Download NLTK data if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stop words
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

XML_FOLDER_PATH = os.path.join(settings.BASE_DIR, 'xml_folder')

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


def generate_word_frequency_chart_index(words_count, xml_filename):
    if not words_count:
        print("words_count is empty. No data to generate chart.")
        return  

    words, counts = zip(*sorted(words_count.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='blue', alpha=0.7, label='Bar Chart')  # 長條圖
    plt.plot(words, counts, color='orange', marker='o', linestyle='-', linewidth=2, label='Line Chart')  # 折線圖
    
    plt.xlabel('Words')
    plt.ylabel('Frequency of words')
    plt.xticks(rotation=45, ha='right')
    plt.title('ZIPF Distribution')
    plt.legend()

    chart_filename = f"word_frequency_{xml_filename}.png"
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    print(chart_path)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close() 
    
    return chart_path

def extract_unique_words_from_titles():
    all_words = set()
    categories = ['enterovirus', 'food_safety']
    for category in categories:
        category_folder = os.path.join(XML_FOLDER_PATH, category)

        if not os.path.exists(category_folder):
            continue

        for file in os.listdir(category_folder):
            if file.endswith(".xml"):
                file_path = os.path.join(category_folder, file)
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

def index_view(request):
    query = request.GET.get('q')
    selected_year = request.GET.get('year')
    files = []
    years = set()
    categories = ['enterovirus', 'food_safety']  # Define your categories
    result_limit = 20  # Maximum number of results to display
    all_words = extract_unique_words_from_titles()
    similar_words = []
    if query:
        similar_words = get_similar_words(query, all_words)

    for category in categories:
        category_folder = os.path.join(XML_FOLDER_PATH, category)

        if not os.path.exists(category_folder):
            continue

        for file in os.listdir(category_folder):
            if file.endswith(".xml"):
                file_path = os.path.join(category_folder, file)
                tree = ET.parse(file_path)
                root = tree.getroot()

                article_title_element = root.find('.//ArticleTitle')
                e_location_id_element = root.find('.//ELocationID[@EIdType="doi"]')
                pmid_element = root.find('.//PMID')
                year_element = root.find('.//PubDate/Year')
                abstract_text_elements = root.findall('.//AbstractText')

                article_title = article_title_element.text if article_title_element is not None else 'N/A'
                doi = e_location_id_element.text if e_location_id_element is not None else 'N/A'
                pmid = pmid_element.text if pmid_element is not None else 'N/A'
                pub_year = year_element.text if year_element is not None else 'N/A'

                abstract_text = ''.join([ET.tostring(el, encoding='unicode') for el in abstract_text_elements])
                abstract_text = re.sub(r'<[^>]+>', '', abstract_text)

                if pub_year != 'N/A':
                    years.add(pub_year)

                if query and ((article_title and query.lower() in article_title.lower()) or (abstract_text and query.lower() in abstract_text.lower())):
                    keyword_count_title = len(re.findall(re.escape(query), article_title or '', re.IGNORECASE))
                    keyword_count_abstract = len(re.findall(re.escape(query), abstract_text or '', re.IGNORECASE))
                    total_keyword_count = keyword_count_title + keyword_count_abstract

                    stat = statistics(abstract_text)

                    if selected_year:
                        if selected_year == pub_year:
                            pass
                        else:
                            continue

                    # Compute top five words
                    tokens = word_tokenize(abstract_text)
                    words = [word.lower() for word in tokens if word.isalpha()]
                    word_counts = Counter(words)
                    top_words = word_counts.most_common(5)

                    # Compute top five words excluding stop words
                    words_no_stop = [word for word in words if word not in stop_words]
                    word_counts_no_stop = Counter(words_no_stop)
                    top_words_no_stop = word_counts_no_stop.most_common(5)

                    # Zip the top words for parallel iteration in the template
                    top_words_zipped = list(zip(top_words, top_words_no_stop))
                    
                    chart_filename = f"figure_{file}.png"
                    chart_path = os.path.join(settings.MEDIA_ROOT, chart_filename)

                    if not os.path.exists(chart_path):
                        generate_word_frequency_chart_index(word_counts, file)

                    files.append({
                        'filename': file,
                        'category': category,
                        'title': highlight_keyword(article_title, query),
                        'abstract': highlight_keyword(abstract_text, query),
                        'doi': doi,
                        'pmid': pmid,
                        'pub_year': pub_year,
                        'keyword_count': total_keyword_count,
                        'statistics': stat,
                        'top_words_zipped': top_words_zipped,
                        'chart': chart_filename,
                    })

                    if len(files) >= result_limit:
                        break

            if len(files) >= result_limit:
                break

    # Sort files based on keyword count
    files = sorted(files, key=lambda x: x['keyword_count'], reverse=True)[:result_limit]

    result_message = f"顯示前 {result_limit} 筆相關結果，若需要完整資料，請精確化搜尋條件。"

    return render(request, 'index.html', {
        'files': files,
        'query': query,
        'years': sorted(years),
        'similar_words': similar_words,
        'selected_year': selected_year,
        'result_message': result_message,
    })

def file_analysis(request, filename):
    # 定義資料夾名稱作為類別標籤
    categories = ['enterovirus', 'food_safety']
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

# Get the maximum frequency for y-axis scaling
def get_max_frequency(counts_list):
    max_values = [max(counts.values()) if counts else 0 for counts in counts_list]
    return max(max_values)

# Generate word frequency chart (line chart, single y-axis)
def generate_word_frequency_chart(words_count, chart_filename, max_frequency):
    if not words_count:
        print("words_count is empty. No data to generate chart.")
        return

    # Sort words by their frequency in descending order
    sorted_words = dict(sorted(words_count.items(), key=lambda x: x[1], reverse=True))
    words, counts = zip(*sorted_words.items()) 

    # Create line chart
    plt.figure(figsize=(14, 6))
    plt.plot(words, counts, color='blue', alpha=0.7, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max_frequency)
    plt.title(f'Word Frequency for {chart_filename}')
    
    # Save chart
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

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

def generate_word_frequency_chart(words_count, chart_filename, top_n=None):
    if not words_count:
        print("words_count is empty. No data to generate chart.")
        return

    # Sort words by their frequency in descending order
    sorted_words = dict(sorted(words_count.items(), key=lambda x: x[1], reverse=True))
    
    # If top_n is specified, only keep the top N words
    if top_n:
        sorted_words = dict(list(sorted_words.items())[:top_n])

    words, counts = zip(*sorted_words.items())
    
    # Find the minimum and maximum frequencies for dynamic y-axis scaling
    min_frequency = min(counts)
    max_frequency = max(counts)

    # Create line chart
    plt.figure(figsize=(14, 6))
    plt.plot(words, counts, color='blue', alpha=0.7, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    
    # Set dynamic y-axis range based on min/max frequency
    plt.ylim(min_frequency, max_frequency + (max_frequency - min_frequency) * 0.1)  # Add a 10% margin to the top
    
    title_part = f'Top {top_n}' if top_n else 'All Words'
    plt.title(f'Word Frequency for {chart_filename} ({title_part})')
    
    # Save chart
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path


def generate_word_frequency_chart_all(words_count, chart_filename, top_n=None):
    if not words_count:
        print("words_count is empty. No data to generate chart.")
        return

    sorted_words = dict(sorted(words_count.items(), key=lambda x: x[1], reverse=True))
    if top_n:
        sorted_words = dict(list(sorted_words.items())[:top_n])

    words, counts = zip(*sorted_words.items())
    min_frequency = min(counts)
    max_frequency = max(counts)

    plt.figure(figsize=(14, 6))
    plt.plot(words, counts, color='blue', alpha=0.7, marker='o', linestyle='-', linewidth=2)
    plt.gca().xaxis.set_visible(False)
    plt.ylim(min_frequency, max_frequency + (max_frequency - min_frequency) * 0.1) 
    
    title_part = f'Top {top_n}' if top_n else 'All Words'
    plt.title(f'Word Frequency for {chart_filename} ({title_part})')
    
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def generate_word_frequency_chart_log(words_count, chart_filename, top_n=None):
    if not words_count:
        print("words_count is empty. No data to generate chart.")
        return

    sorted_words = dict(sorted(words_count.items(), key=lambda x: x[1], reverse=True))
    if top_n:
        sorted_words = dict(list(sorted_words.items())[:top_n])

    words, counts = zip(*sorted_words.items())
    counts = np.log(np.array(counts) + 1)
    min_frequency = min(counts)
    max_frequency = max(counts)

    plt.figure(figsize=(14, 6))
    plt.plot(words, counts, color='blue', alpha=0.7, marker='o', linestyle='-', linewidth=2)
    plt.gca().xaxis.set_visible(False)
    plt.ylim(min_frequency, max_frequency + (max_frequency - min_frequency) * 0.1) 
    
    title_part = f'Top {top_n}' if top_n else 'All Words'
    plt.title(f'Word Frequency for {chart_filename} ({title_part})')
    
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def generate_word_frequency_chart_logxy(words_count, chart_filename, top_n=None):
    if not words_count:
        print("words_count is empty. No data to generate chart.")
        return

    sorted_words = dict(sorted(words_count.items(), key=lambda x: x[1], reverse=True))
    if top_n:
        sorted_words = dict(list(sorted_words.items())[:top_n])

    words, counts = zip(*sorted_words.items())
    counts = np.log(np.array(counts) + 1)
    x_values = np.arange(1, len(words) + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(x_values, counts, color='blue', alpha=0.7, marker='o', linestyle='-', linewidth=2)
    plt.xscale('log')

    plt.xticks([10**i for i in range(int(np.log10(len(words))) + 1)], 
               [f'$10^{i}$' for i in range(int(np.log10(len(words))) + 1)])

    min_frequency = min(counts)
    max_frequency = max(counts)
    plt.ylim(min_frequency, max_frequency + (max_frequency - min_frequency) * 0.1) 
    
    title_part = f'Top {top_n}' if top_n else 'All Words'
    plt.title(f'Word Frequency for {chart_filename} ({title_part})')
    
    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def generate_word_cloud(word_counts, chart_filename):
    if not word_counts:
        print("word_counts is empty. No data to generate word cloud.")
        return

    font_path = 'C:/Windows/Fonts/Arial.ttf'  # Windows 路徑，確認這個字型存在
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    chart_path = os.path.join(settings.MEDIA_ROOT, 'static', 'charts', chart_filename)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# Distribution view to generate charts (all words and top 20 words)
def distribution_view(request):
    folders = ['food_safety', 'enterovirus']
    chart_paths = {}

    for folder in folders:
        folder_path = os.path.join(settings.MEDIA_ROOT, 'xml_folder', folder)
        word_counts = get_word_counts_from_folder(folder_path)

        # 1. Generate original word frequency chart (All words)
        chart_filename_all = f'word_frequency_{folder}_all.png'
        if not check_chart_exists(chart_filename_all):
            generate_word_frequency_chart_all(word_counts, chart_filename_all)
        chart_paths[f'{folder}_all'] = f'static/charts/{chart_filename_all}'

        # 2. Generate word frequency chart (Top 20 words)
        chart_filename_top20 = f'word_frequency_{folder}_top20.png'
        if not check_chart_exists(chart_filename_top20):
            generate_word_frequency_chart(word_counts, chart_filename_top20, top_n=20)
        chart_paths[f'{folder}_top20'] = f'static/charts/{chart_filename_top20}'

        # Generate word frequency chart (Top 50 words)
        chart_filename_top50 = f'word_frequency_{folder}_top50.png'
        if not check_chart_exists(chart_filename_top50):
            generate_word_frequency_chart(word_counts, chart_filename_top50, top_n=50)
        chart_paths[f'{folder}_top50'] = f'static/charts/{chart_filename_top50}'

        chart_filename_top100 = f'word_frequency_{folder}_top100.png'
        if not check_chart_exists(chart_filename_top100):
            generate_word_frequency_chart(word_counts, chart_filename_top100, top_n=100)
        chart_paths[f'{folder}_top100'] = f'static/charts/{chart_filename_top100}'

        chart_filename_log = f'word_frequency_{folder}_log.png'
        if not check_chart_exists(chart_filename_log):
            generate_word_frequency_chart_log(word_counts, chart_filename_log)
        chart_paths[f'{folder}_log'] = f'static/charts/{chart_filename_log}'

        chart_filename_log = f'word_frequency_{folder}_logxy.png'
        if not check_chart_exists(chart_filename_log):
            generate_word_frequency_chart_logxy(word_counts, chart_filename_log)
        chart_paths[f'{folder}_log'] = f'static/charts/{chart_filename_log}'

        # 3. Generate Porter stemmed chart (All words)
        stemmed_counts = apply_stemming(word_counts)
        stemmed_chart_filename_all = f'word_frequency_stemmed_{folder}_all.png'
        if not check_chart_exists(stemmed_chart_filename_all):
            generate_word_frequency_chart_all(stemmed_counts, stemmed_chart_filename_all)
        chart_paths[f'{folder}_stemmed_all'] = f'static/charts/{stemmed_chart_filename_all}'

        # 4. Generate Porter stemmed chart (Top 20 words)
        stemmed_chart_filename_top20 = f'word_frequency_stemmed_{folder}_top20.png'
        if not check_chart_exists(stemmed_chart_filename_top20):
            generate_word_frequency_chart(stemmed_counts, stemmed_chart_filename_top20, top_n=20)
        chart_paths[f'{folder}_stemmed_top20'] = f'static/charts/{stemmed_chart_filename_top20}'

        stemmed_chart_filename_top50 = f'word_frequency_stemmed_{folder}_top50.png'
        if not check_chart_exists(stemmed_chart_filename_top50):
            generate_word_frequency_chart(stemmed_counts, stemmed_chart_filename_top50, top_n=50)
        chart_paths[f'{folder}_stemmed_top50'] = f'static/charts/{stemmed_chart_filename_top50}'

        stemmed_chart_filename_top100 = f'word_frequency_stemmed_{folder}_top100.png'
        if not check_chart_exists(stemmed_chart_filename_top100):
            generate_word_frequency_chart(stemmed_counts, stemmed_chart_filename_top100, top_n=100)
        chart_paths[f'{folder}_stemmed_top100'] = f'static/charts/{stemmed_chart_filename_top100}'

        # 5. Generate stop word filtered chart (All words)
        filtered_counts = remove_stopwords(word_counts)
        filtered_chart_filename_all = f'word_frequency_filtered_{folder}_all.png'
        if not check_chart_exists(filtered_chart_filename_all):
            generate_word_frequency_chart_all(filtered_counts, filtered_chart_filename_all)
        chart_paths[f'{folder}_filtered_all'] = f'static/charts/{filtered_chart_filename_all}'

        # 6. Generate stop word filtered chart (Top 20 words)
        filtered_chart_filename_top20 = f'word_frequency_filtered_{folder}_top20.png'
        if not check_chart_exists(filtered_chart_filename_top20):
            generate_word_frequency_chart(filtered_counts, filtered_chart_filename_top20, top_n=20)
        chart_paths[f'{folder}_filtered_top20'] = f'static/charts/{filtered_chart_filename_top20}'

        filtered_chart_filename_top50 = f'word_frequency_filtered_{folder}_top50.png'
        if not check_chart_exists(filtered_chart_filename_top50):
            generate_word_frequency_chart(filtered_counts, filtered_chart_filename_top50, top_n=50)
        chart_paths[f'{folder}_filtered_top50'] = f'static/charts/{filtered_chart_filename_top50}'

        filtered_chart_filename_top100 = f'word_frequency_filtered_{folder}_top100.png'
        if not check_chart_exists(filtered_chart_filename_top100):
            generate_word_frequency_chart(filtered_counts, filtered_chart_filename_top100, top_n=100)
        chart_paths[f'{folder}_filtered_top100'] = f'static/charts/{filtered_chart_filename_top100}'

        chart_filename_wordcloud = f'wordcloud_{folder}.png'
        if not check_chart_exists(chart_filename_wordcloud):
            generate_word_cloud(word_counts, chart_filename_wordcloud)
        chart_paths[f'{folder}_wordcloud'] = f'static/charts/{chart_filename_wordcloud}'

    return render(request, 'distribution.html', chart_paths)


def word_view(request):
    folders = ['food_safety', 'enterovirus']
    results = {}

    for folder in folders:
        folder_path = os.path.join(settings.MEDIA_ROOT, 'xml_folder', folder)
        original_word_counts = get_word_counts_from_folder(folder_path)
        porter_stemmed_counts = apply_stemming(original_word_counts)
        stop_word_filtered_counts = remove_stopwords(original_word_counts)

        # 根據頻率從高到低排序單詞
        original_words_sorted = sorted(original_word_counts.items(), key=lambda x: x[1], reverse=True)
        porter_stemmed_words_sorted = sorted(porter_stemmed_counts.items(), key=lambda x: x[1], reverse=True)
        stop_word_filtered_words_sorted = sorted(stop_word_filtered_counts.items(), key=lambda x: x[1], reverse=True)

        # 獲取前20%到40%的單詞
        def get_percentage_range(sorted_counts):
            total_count = len(sorted_counts)
            start_index = int(total_count * 0.2)
            end_index = int(total_count * 0.4)
            return sorted_counts[start_index:end_index]

        original_words_range = get_percentage_range(original_words_sorted)
        porter_stemmed_words_range = get_percentage_range(porter_stemmed_words_sorted)
        stop_word_filtered_words_range = get_percentage_range(stop_word_filtered_words_sorted)

        # 提取單詞和頻率並轉換為字符串顯示
        original_words = '\n'.join(f"{word}: {count}" for word, count in original_words_range)
        porter_stemmed_words = '\n'.join(f"{word}: {count}" for word, count in porter_stemmed_words_range)
        stop_word_filtered_words = '\n'.join(f"{word}: {count}" for word, count in stop_word_filtered_words_range)

        results[folder] = {
            'original_words': original_words,
            'porter_stemmed_words': porter_stemmed_words,
            'stop_word_filtered_words': stop_word_filtered_words
        }

    context = {
        'food_safety': results['food_safety'],
        'enterovirus': results['enterovirus']
    }

    return render(request, 'count.html', context)