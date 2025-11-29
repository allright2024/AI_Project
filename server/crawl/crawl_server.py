import asyncio
import subprocess
import os
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import uvicorn

app = FastAPI()
scheduler = AsyncIOScheduler()

from llm_processor import process_new_posts
from embedding_processor import run_embedding_pipeline

DIR = os.path.dirname(os.path.abspath(__file__))
CRAWLERS = [
    "crawl_dup.py",
    "crawl_chemistry_engineering.py",
    "crawl_env_engineering.py"
]

def run_crawler(script_name):
    """Runs a single crawler script using subprocess."""
    script_path = os.path.join(DIR, script_name)
    print(f"[{datetime.now()}] Starting crawler: {script_name}")
    try:
        # Use the same python executable as the server
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"[{datetime.now()}] Finished crawler: {script_name}")
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Error running crawler: {script_name}")
        print(f"Error Output:\n{e.stderr}")

async def run_all_crawlers():
    """Runs all registered crawlers sequentially."""
    print(f"[{datetime.now()}] Scheduled crawl started.")
    for crawler in CRAWLERS:
        # Run in a separate thread to avoid blocking the event loop
        await asyncio.to_thread(run_crawler, crawler)
    
    # Run LLM processing after crawling
    print(f"[{datetime.now()}] Starting LLM processing...")
    any_updated = await asyncio.to_thread(process_new_posts)
    
    if any_updated:
        print(f"[{datetime.now()}] New posts processed. Starting embedding pipeline...")
        await asyncio.to_thread(run_embedding_pipeline)
    else:
        print(f"[{datetime.now()}] No new posts processed. Skipping embedding pipeline.")
    
    print(f"[{datetime.now()}] Scheduled crawl finished.")

@app.on_event("startup")
async def startup_event():
    """Start the scheduler on app startup."""
    scheduler.add_job(
        run_all_crawlers,
        trigger=IntervalTrigger(hours=3),
        id="crawl_job",
        name="Run all crawlers every 6 hours",
        replace_existing=True,
    )
    scheduler.start()
    print("Scheduler started. Crawlers will run every 6 hours.")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the scheduler on app shutdown."""
    scheduler.shutdown()
    print("Scheduler shutdown.")

@app.post("/crawl")
async def trigger_crawl(background_tasks: BackgroundTasks):
    """Manually trigger all crawlers in the background."""
    background_tasks.add_task(run_all_crawlers)
    return {"message": "Crawl triggered in background"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    jobs = scheduler.get_jobs()
    job_info = [{"id": job.id, "next_run_time": job.next_run_time} for job in jobs]
    return {"status": "ok", "scheduled_jobs": job_info}

if __name__ == "__main__":
    uvicorn.run("crawl_server:app", host="0.0.0.0", port=8000, reload=True)
