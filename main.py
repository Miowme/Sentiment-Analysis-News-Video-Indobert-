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

client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client.thesis_sentiment 
raw_collection = db.raw_comments
analysis_collection = db.analyzed_results

app = FastAPI(title="YouTube Sentiment API (Official v3)")

# Konfigurasi YouTube API
DEVELOPER_KEY = "AIzaSyB0NqKUD6O-P9dmNXx_klgZDkCHzWfPmI0" 
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

# Konfigurasi IndoBERT
PRETRAINED_MODEL = "mdhugol/indonesia-bert-sentiment-classification"
LABEL_INDEX = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
classifier = None

def get_video_id(url: str):
    """Mengekstrak ID video dari berbagai format URL YouTube"""
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
        raise HTTPException(status_code=400, detail="ID Video tidak ditemukan dalam URL")

    try:
        comments = []
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
                comments.append(comment_text)

            next_page_token = response.get('nextPageToken')

            if not next_page_token:
                break

        if not comments:
            return {"status": "Empty", "message": "Tidak ada komentar ditemukan atau dinonaktifkan."}

        document = {
            "video_id": video_id,
            "url": request.url,
            "total_scraped": len(comments),
            "comments": comments,
            "source": "YouTube API v3",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        await raw_collection.insert_one(document)
        if "_id" in document: del document["_id"]

        return {"status": "Success", "data": document}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube API Error: {str(e)}")

@app.post("/analyze")
async def analyze_sentiment(request: URLRequest):
    video_id = get_video_id(request.url)

    raw_data = await raw_collection.find_one({"video_id": video_id})
    if not raw_data:
        raise HTTPException(status_code=404, detail="Data belum di-scrape. Jalankan /scrapping dulu.")

    if classifier is None:
        raise HTTPException(status_code=500, detail="Model IndoBERT belum dimuat/siap.")

    texts = raw_data["comments"]
    processed = []
    labels = []

    batch_size = 32  
    
    print(f"--- Mulai Analisis {len(texts)} komentar... ---")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        batch_results = await asyncio.to_thread(classifier, batch_texts)

        for text, res in zip(batch_texts, batch_results):
            label = LABEL_INDEX.get(res['label'], "unknown")
            labels.append(label)
            processed.append({
                "text": text, 
                "label": label, 
                "confidence": f"{res['score']*100:.2f}%"
            })

    counts = Counter(labels)

    overall_sentiment = "N/A"
    if counts:
        overall_sentiment = counts.most_common(1)[0][0]

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
    
    if "_id" in analysis_doc: del analysis_doc["_id"]

    print(f"--- Analisis Selesai untuk Video: {video_id} ---")
    return analysis_doc

@app.get("/all-results")
async def get_results(limit: int = 10):
    try:
        cursor = analysis_collection.find().sort("analyzed_at", -1).limit(limit)
        results = await cursor.to_list(length=limit) 

        for res in results:

            res["_id"] = str(res["_id"])
      
        return {
            "status": "success",
            "total_stored": len(results), 
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data: {str(e)}")