import os
from django.conf import settings
from django.shortcuts import render, redirect
import xml.etree.ElementTree as ET
from django.utils.html import mark_safe
from django.core.files.storage import FileSystemStorage
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from django.http import JsonResponse
from django.contrib import messages

nltk.download('stopwords')
nltk.download('punkt_tab')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

XML_FOLDER_PATH = os.path.join(settings.BASE_DIR, 'xml_folder')

def highlight_keyword(text, query):
    if query and text:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
        return mark_safe(highlighted)
    return text

def highlight_text(text, query):
    if query and text:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
        return mark_safe(highlighted)
    return text

def count_sentences(text):
    abbreviations = r'\b(?:U\.S\.|e\.g\.|i\.e\.|Dr\.|Mr\.|Ms\.|Prof\.|Ltd\.|Inc\.|Jr\.|Sr\.)\b'
    abbreviation_pattern = re.compile(abbreviations, re.IGNORECASE)
    sentence_endings = re.finditer(r'[.!?]', text)
    
    if text == "":
        return 0
    
    sentence_count = 0
    # last_position = 0
    
    for match in sentence_endings:
        end_pos = match.start()
        
        if end_pos + 1 < len(text) and text[end_pos + 1] == ' ' and end_pos + 2 < len(text) and (text[end_pos + 2].isupper() or text[end_pos + 2].isdigit()):
            before_punctuation = text[:end_pos].strip().split()[-1] if text[:end_pos].strip() else ''
            
            if not abbreviation_pattern.search(before_punctuation):
                sentence_count += 1
                # last_position = end_pos + 1

    # if text[last_position:].strip() and text.strip()[-1] in ['.', '?', '!']:
        # sentence_count += 1

    sentence_count += 1
    return sentence_count

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def count_letters_spaces(text):
    letters = sum(1 for char in text if char.isalpha())
    return letters

def count_stemmed_words(text):
    words = re.findall(r'\b\w+\b', text)
    stemmed_words = [ps.stem(word.lower()) for word in words]
    return len(set(stemmed_words))

def count_stop_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_word_count = sum(1 for word in words if word in stop_words)
    return stop_word_count

def count_ascii_non_ascii(text):
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return ascii_count, non_ascii_count

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'xml_folder'))
        fs.save(uploaded_file.name, uploaded_file)
        messages.success(request, '文件上傳成功！')
        return redirect('index_view')
    return render(request, 'index.html')

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

def index_view(request):
    query = request.GET.get('q')
    selected_year = request.GET.get('year')
    files = []
    years = set()
    xml_folder = os.path.join(settings.BASE_DIR, 'xml_folder')

    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join(xml_folder, file))
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

            # Join abstract text content
            abstract_text = ''.join([ET.tostring(el, encoding='unicode') for el in abstract_text_elements])
            abstract_text = re.sub(r'<[^>]+>', '', abstract_text)  # Remove any XML tags from the content

            # Add available year to the years set if it is not 'N/A'
            if pub_year != 'N/A':
                years.add(pub_year)

            # Filter by query: search in both title and abstract
            if query and (query.lower() in article_title.lower() or query.lower() in abstract_text.lower()):
                # Count keyword occurrences
                keyword_count_title = len(re.findall(re.escape(query), article_title, re.IGNORECASE))
                keyword_count_abstract = len(re.findall(re.escape(query), abstract_text, re.IGNORECASE))
                total_keyword_count = keyword_count_title + keyword_count_abstract

                stat = statistics(abstract_text)

                # Filter by selected year if provided, or include 'N/A' years if no year is selected
                if selected_year:
                    if selected_year == pub_year:
                        files.append({
                            'filename': file,
                            'title': highlight_keyword(article_title, query),
                            'abstract': highlight_keyword(abstract_text, query),
                            'doi': doi,
                            'pmid': pmid,
                            'pub_year': pub_year,
                            'keyword_count': total_keyword_count,
                            'statistics': stat
                        })
                else:  # No year selected, include all files including those with 'N/A' year
                    files.append({
                        'filename': file,
                        'title': highlight_keyword(article_title, query),
                        'abstract': highlight_keyword(abstract_text, query),
                        'doi': doi,
                        'pmid': pmid,
                        'pub_year': pub_year,
                        'keyword_count': total_keyword_count,
                        'statistics': stat
                    })

    files.sort(key=lambda x: x['keyword_count'], reverse=True)

    return render(request, 'index.html', {
        'files': files,
        'query': query,
        'years': sorted(years),
        'selected_year': selected_year,
    })

def update_searchable_files(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('selected_files')
        request.session['searchable_files'] = selected_files
    return redirect('dataset_view')

def delete_file(request, filename):
    xml_folder = os.path.join(settings.BASE_DIR, 'xml_folder')
    file_path = os.path.join(xml_folder, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return JsonResponse({'status': 'success', 'message': 'File deleted successfully.'})
    return JsonResponse({'status': 'error', 'message': 'File not found.'})

def file_analysis(request, filename):
    xml_folder = os.path.join(settings.BASE_DIR, 'xml_folder')
    file_path = os.path.join(xml_folder, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

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


def dataset_view(request):
    xml_folder = os.path.join(settings.BASE_DIR, 'xml_folder')
    files = []

    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            files.append(file)

    return render(request, 'dataset.html', {'files': files})

def analyze_content(content):
    letters = count_letters_spaces(content)
    stemmed_words_count = count_stemmed_words(content)
    stop_word_count = count_stop_words(content) 
    ascii_count, non_ascii_count = count_ascii_non_ascii(content)

    return {
        'characters': letters,
        'words': stemmed_words_count,
        'sentences': stop_word_count,
        'ascii_count': ascii_count,
        'non_ascii_count': non_ascii_count
    }

def extract_info(tree):
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

    return article_title, doi, pmid, pub_year, abstract_text

def count_keyword_occurrences(text, keyword):
    if not keyword:
        return 0
    return len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))

def clear_session(request):
    request.session.flush()
    return redirect('compare_view')

def compare_view(request):
    if request.method == 'POST':
        file1 = request.FILES.get('file1')
        file2 = request.FILES.get('file2')

        if not file1 or not file2:
            return render(request, 'compare.html', {'error': 'Please upload two XML files.'})

        fs = FileSystemStorage()
        file1_path = fs.save(file1.name, file1)
        file2_path = fs.save(file2.name, file2)
        file1_content = fs.open(file1_path).read().decode('utf-8')
        file2_content = fs.open(file2_path).read().decode('utf-8')

        try:
            tree1 = ET.ElementTree(ET.fromstring(file1_content))
            tree2 = ET.ElementTree(ET.fromstring(file2_content))

            title1, doi1, pmid1, year1, abstract1 = extract_info(tree1)
            title2, doi2, pmid2, year2, abstract2 = extract_info(tree2)

            request.session['file1_data'] = {
                'filename': file1.name,
                'title': title1,
                'abstract': abstract1,
                'doi': doi1,
                'pmid': pmid1,
                'pub_year': year1
            }
            request.session['file2_data'] = {
                'filename': file2.name,
                'title': title2,
                'abstract': abstract2,
                'doi': doi2,
                'pmid': pmid2,
                'pub_year': year2
            }

            stats1 = analyze_content(abstract1)
            stats2 = analyze_content(abstract2)
            request.session['file1_data']['statistics'] = stats1
            request.session['file2_data']['statistics'] = stats2

            return redirect('compare_view')

        except ET.ParseError:
            return render(request, 'compare.html', {'error': 'Error parsing XML files.'})

    elif 'file1_data' in request.session and 'file2_data' in request.session:
        file1_data = request.session['file1_data']
        file2_data = request.session['file2_data']
        keyword = request.GET.get('keyword', '')

        if keyword:
            file1_data['title'] = highlight_keyword(file1_data['title'], keyword)
            file1_data['abstract'] = highlight_keyword(file1_data['abstract'], keyword)
            file1_data['keyword_count'] = count_keyword_occurrences(file1_data['title'], keyword) + count_keyword_occurrences(file1_data['abstract'], keyword)

            file2_data['title'] = highlight_keyword(file2_data['title'], keyword)
            file2_data['abstract'] = highlight_keyword(file2_data['abstract'], keyword)
            file2_data['keyword_count'] = count_keyword_occurrences(file2_data['title'], keyword) + count_keyword_occurrences(file2_data['abstract'], keyword)

        context = {
            'file1': file1_data,
            'file2': file2_data,
            'keyword': keyword
        }

        return render(request, 'compare.html', context)

    return render(request, 'compare.html')