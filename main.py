import motor.motor_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from collections import Counter 
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import asyncio
import googleapiclient.discovery
import re
import time
import unicodedata

client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client.thesis_sentiment 
raw_collection = db.raw_comments
analysis_collection = db.analyzed_results

app = FastAPI(title="YouTube Sentiment API (Clean Storage Version)")

# Konfigurasi YouTube API
DEVELOPER_KEY = "AIzaSyB0NqKUD6O-P9dmNXx_klgZDkCHzWfPmI0" 
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

# Konfigurasi IndoBERT
PRETRAINED_MODEL = "mdhugol/indonesia-bert-sentiment-classification"
LABEL_INDEX = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
classifier = None

def clean_and_normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.replace('ß', 's') 
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = " ".join(text.lower().split())
    return text

def get_video_id(url: str):
    reg = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(reg, url)
    if match:
        return match.group(1)
    return None

def load_model():
    global classifier
    print("--- Memuat Model IndoBERT... ---")
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("--- Model Siap! ---")

@app.on_event("startup")
async def startup():
    asyncio.create_task(asyncio.to_thread(load_model))

class URLRequest(BaseModel):
    url: str

@app.post("/scrapping")
async def scrape_youtube_api(request: URLRequest):
    video_id = get_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="ID Video tidak ditemukan")

    try:
        cleaned_comments = [] 
        next_page_token = None

        while True:
            youtube_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100, 
                textFormat="plainText",
                pageToken=next_page_token 
            )
            response = youtube_request.execute()

            for item in response.get('items', []):
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                cleaned_text = clean_and_normalize(comment_text)

                if cleaned_text:
                    cleaned_comments.append(cleaned_text)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        if not cleaned_comments:
            return {"status": "Empty", "message": "Tidak ada komentar valid."}

        document = {
            "video_id": video_id,
            "url": request.url,
            "total_scraped": len(cleaned_comments),
            "comments": cleaned_comments, 
            "source": "YouTube API v3",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        await raw_collection.update_one(
            {"video_id": video_id},
            {"$set": document},
            upsert=True
        )
        
        return {
            "status": "Success",
            "message": f"Berhasil mengambil {len(cleaned_comments)} komentar.",
            "data": document 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

@app.post("/analyze")
async def analyze_sentiment(request: URLRequest):
    video_id = get_video_id(request.url)
    raw_data = await raw_collection.find_one({"video_id": video_id})
    
    if not raw_data:
        raise HTTPException(status_code=404, detail="Data belum di-scrape.")

    if classifier is None:
        raise HTTPException(status_code=500, detail="Model belum siap.")

    cleaned_texts = raw_data.get("comments", [])
    processed = []
    labels = []
    batch_size = 32 
    
    print(f"--- Menganalisis {len(cleaned_texts)} komentar ---")

    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i : i + batch_size]

        valid_batch = [t if t.strip() else "netral" for t in batch]

        batch_results = await asyncio.to_thread(
            classifier, 
            valid_batch, 
            truncation=True,    
            max_length=512     
        )

        for text, res in zip(batch, batch_results):
            label = LABEL_INDEX.get(res['label'], "unknown")
            labels.append(label)
            processed.append({
                "text": text, 
                "label": label, 
                "confidence": f"{res['score']*100:.2f}%"
            })

    counts = Counter(labels)
    overall_sentiment = counts.most_common(1)[0][0] if counts else "N/A"

    analysis_doc = {
        "video_id": video_id,
        "summary": {
            "total": len(processed),
            "overall": overall_sentiment,
            "counts": dict(counts)
        },
        "results": processed,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    await analysis_collection.update_one(
        {"video_id": video_id},
        {"$set": analysis_doc},
        upsert=True
    )
    
    return analysis_doc

@app.get("/all-results")
async def get_results(limit: int = 10):
    cursor = analysis_collection.find().sort("analyzed_at", -1).limit(limit)
    results = await cursor.to_list(length=limit) 
    for res in results:
        res["_id"] = str(res["_id"])
    return {"data": results}