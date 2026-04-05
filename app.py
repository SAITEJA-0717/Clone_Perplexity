import os
import math
import re
import asyncio
import datetime
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import PyPDF2

# Web Search and Crawling imports
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the Gemini client using the API key from .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Conversation history for context (last 7 exchanges)
conversation_history = []
MAX_HISTORY = 7

# ===== RAG Document Store =====
rag_store = {
    'filename': None,
    'chunks': [],
    'tfidf': [],
    'vocab': {},
    'idf': {}
}

# ===== CRAWL4AI WEB SEARCH =====

async def search_and_crawl(query, num_results=10):
    """Perform a web search for the query, extract the top URLs, and crawl them."""
    ddgs = DDGS()
    try:
        # Perform search
        results = list(ddgs.text(query, max_results=num_results))
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        results = []
    
    if not results:
        return []
    
    urls = [r['href'] for r in results]
    sources = []
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Create asynchronous crawl tasks for all URLs
        tasks = [crawler.arun(url=url) for url in urls]
        crawl_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, res in enumerate(crawl_results):
            if not isinstance(res, Exception) and getattr(res, 'success', False):
                # Truncate markdown to ~15000 chars for deeper scraping
                markdown_snippet = res.markdown[:15000] if res.markdown else ""
                if markdown_snippet:
                    sources.append({
                        'title': results[i]['title'],
                        'url': urls[i],
                        'snippet': results[i]['body'],
                        'content': markdown_snippet
                    })
            else:
                # Fallback to duckduckgo snippet if crawl fails
                sources.append({
                    'title': results[i]['title'],
                    'url': urls[i],
                    'snippet': results[i]['body'],
                    'content': results[i]['body']
                })
    return sources

# ===== RAG Helper Functions =====
# ... (same as before)

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def compute_tf(tokens):
    counter = Counter(tokens)
    total = len(tokens)
    if total == 0: return {}
    return {term: count / total for term, count in counter.items()}

def compute_idf(documents_tokens):
    n = len(documents_tokens)
    idf = {}
    all_terms = set()
    for doc_tokens in documents_tokens:
        all_terms.update(set(doc_tokens))
    for term in all_terms:
        doc_count = sum(1 for doc_tokens in documents_tokens if term in set(doc_tokens))
        idf[term] = math.log((n + 1) / (doc_count + 1)) + 1
    return idf

def build_tfidf_vectors(chunks):
    documents_tokens = [tokenize(chunk) for chunk in chunks]
    idf = compute_idf(documents_tokens)
    vocab = {term: idx for idx, term in enumerate(sorted(idf.keys()))}
    vectors = []
    for doc_tokens in documents_tokens:
        tf = compute_tf(doc_tokens)
        vector = {}
        for term, tf_val in tf.items():
            if term in vocab:
                vector[vocab[term]] = tf_val * idf.get(term, 0)
        vectors.append(vector)
    return vocab, idf, vectors

def cosine_similarity(vec_a, vec_b):
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0: return 0.0
    return dot_product / (mag_a * mag_b)

def query_to_vector(query, vocab, idf):
    tokens = tokenize(query)
    tf = compute_tf(tokens)
    vector = {}
    for term, tf_val in tf.items():
        if term in vocab:
            vector[vocab[term]] = tf_val * idf.get(term, 0)
    return vector

def retrieve_relevant_chunks(query, top_k=5):
    if not rag_store['chunks']: return []
    query_vec = query_to_vector(query, rag_store['vocab'], rag_store['idf'])
    scored = [(cosine_similarity(query_vec, chunk_vec), i) for i, chunk_vec in enumerate(rag_store['tfidf'])]
    scored.sort(reverse=True, key=lambda x: x[0])
    results = [rag_store['chunks'][idx] for score, idx in scored[:top_k] if score > 0]
    return results

# ===== Routes =====

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global rag_store
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.pdf'): return jsonify({'error': 'Only PDF files are supported'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        text = extract_text_from_pdf(filepath)
        if not text.strip(): return jsonify({'error': 'Could not extract text.'}), 400
        chunks = chunk_text(text, chunk_size=400, overlap=80)
        vocab, idf, vectors = build_tfidf_vectors(chunks)
        rag_store = {
            'filename': file.filename,
            'chunks': chunks,
            'tfidf': vectors,
            'vocab': vocab,
            'idf': idf
        }
        return jsonify({'status': 'success', 'filename': file.filename, 'chunks': len(chunks), 'pages': len(PyPDF2.PdfReader(filepath).pages)})
    except Exception as e:
        return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500

@app.route('/remove-file', methods=['POST'])
def remove_file():
    global rag_store
    rag_store = {'filename': None, 'chunks': [], 'tfidf': [], 'vocab': {}, 'idf': {}}
    return jsonify({'status': 'removed'})

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history

    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    use_web_search = data.get('use_web_search', True)
    entrepreneur_mode = data.get('entrepreneur_mode', False)
    
    context_type = "None"
    injected_context = ""
    sources_metadata = []

    # 1. RAG Check
    if rag_store['chunks']:
        relevant_chunks = retrieve_relevant_chunks(user_message, top_k=5)
        if relevant_chunks:
            context_type = "Document"
            merged_chunks = "\n\n---\n\n".join(relevant_chunks)
            injected_context = f"""
The user uploaded a document titled "{rag_store['filename']}". Use the following sections to answer accurately. 
Cite the document if used.

--- DOCUMENT CONTEXT ---
{merged_chunks}
--- END DOCUMENT CONTEXT ---
"""

    # 2. Web Search Check (If no RAG context and web search is enabled)
    elif use_web_search:
        # Run asynchronous search & crawl inside standard Flask sync route with asyncio.run
        sources = asyncio.run(search_and_crawl(user_message, num_results=10))
        if sources:
            context_type = "Web"
            sources_metadata = [{'title': s['title'], 'url': s['url'], 'snippet': s['snippet']} for s in sources]
            
            merged_results = ""
            for i, s in enumerate(sources):
                merged_results += f"\nSOURCE [{i+1}] (Title: {s['title']}, URL: {s['url']}):\n{s['content']}\n---"

            if entrepreneur_mode:
                injected_context = f"""
Below is real-time market data retrieved from the web related to the user's idea. 
Use this exact information to ground your business plan (e.g., market size, competitors). You MUST synthesize this with your own elite AI knowledge to complete the full business plan and investor matching.
Cite the sources when you state facts like market size or competitors, e.g., [1], [2].

--- WEB CONTEXT ---
{merged_results}
--- END WEB CONTEXT ---
"""
            else:
                injected_context = f"""
Below are real-time search results retrieved from the web (via Crawl4AI) related to the user's query. 

CRITICAL INSTRUCTIONS:
1. You MUST use ONLY this exact information to construct your answer.
2. DO NOT hallucinate, invent, or guess any information, including match outcomes, scorecards, or future events.
3. If the provided context does not contain the answer, you must explicitly state: "I cannot find the answer based on the real-time web search results."
4. Cite the sources using their numbers, e.g., [1], [2].

--- WEB CONTEXT ---
{merged_results}
--- END WEB CONTEXT ---
"""

    # Build Prompt
    today_str = datetime.date.today().strftime("%B %d, %Y")
    
    if entrepreneur_mode:
        system_instruction = f"""You are Perplexity Entrepreneur, an elite AI business consultant and startup expert.
Today's date is {today_str}.

When given a brief business idea by the user, you MUST generate a simple, actionable business plan to execute it based on the MVP framework, followed by an 'Investor & Funding Matcher'.
Base your data on the live web context if available, otherwise synthesize realistic estimations.

Use the following exact sections with Markdown headers:
## 1. Executive Summary
## 2. Problem & Solution
## 3. Target Market
## 4. Revenue Model
## 5. Go-To-Market Strategy
## 6. Competitive Landscape
## 7. Team Requirements
## 8. Financial Projections
## 9. Risk Assessment

After the Business Plan, provide the Investor portion:
## 10. Investor & Funding Matcher
Create a detailed Markdown Table that surfaces matched funding options based on the idea. Use these exact columns:
| Funding Type | Recommended Entities | Fit Score (1-10) | Reasoning |
CRITICAL INVESTOR INSTRUCTION:
- You MUST ONLY suggest trusted, legally registered investors.
- Only provide real, well-known VC firms (e.g., Sequoia, Y Combinator, a16z), verified angel networks, or established government grants that are known to invest in this specific industry.
- DO NOT hallucinate fake investor names or funds under any circumstances.

Finally, write a brief, professional **Cold Email Draft** targeted at one of the trusted VC firms you recommended.
"""
    else:
        system_instruction = f"You are Perplexity, a helpful AI assistant connected to real-time RAG and web crawling capabilities. Provide clear, accurate, and well-structured answers using markdown. Cite your sources gracefully. Your knowledge cutoff is {today_str}. Today's date is {today_str}. NEVER hallucinate information if it is not present in the provided context."
    if injected_context:
        system_instruction += f"\n\n{injected_context}"

    # Build history
    contents = []
    for entry in conversation_history:
        contents.append(types.Content(role='user', parts=[types.Part.from_text(text=entry['user'])]))
        contents.append(types.Content(role='model', parts=[types.Part.from_text(text=entry['assistant'])]))

    contents.append(types.Content(role='user', parts=[types.Part.from_text(text=user_message)]))

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=8192,
                system_instruction=system_instruction
            )
        )

        assistant_reply = response.text

        conversation_history.append({'user': user_message, 'assistant': assistant_reply})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]

        return jsonify({
            'response': assistant_reply,
            'context_type': context_type,
            'sources': sources_metadata
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    global conversation_history, rag_store
    conversation_history = []
    rag_store = {'filename': None, 'chunks': [], 'tfidf': [], 'vocab': {}, 'idf': {}}
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
