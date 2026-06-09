"""
Smart Waste Disposal System — FastAPI service.
Target: 500 classifications/hr.
Run: uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
"""
import asyncio,logging,time,os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI,UploadFile,File,HTTPException,BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
RESNET_PATH="models/resnet50_waste.pt"; OUTPUT_PATH="models/resnet50_waste_v2.pt"

clf=None; _sensor_cache={}; _metrics={"total":0,"flagged":0,"latency_sum":0.0,"start_time":time.time()}

from routes.route_optimizer import RouteOptimizer,Bin
from feedback.feedback import FeedbackStore,FineTuner,FineTuneScheduler
store=FeedbackStore()

DEMO_BINS=[Bin("bin_A",51.507,-0.127,87,"A"),Bin("bin_B",51.510,-0.130,41,"B"),
           Bin("bin_C",51.514,-0.125,73,"C"),Bin("bin_D",51.518,-0.118,91,"D"),
           Bin("bin_E",51.505,-0.115,28,"E"),Bin("bin_F",51.512,-0.135,62,"F")]
DEPOT=Bin("DEPOT",51.500,-0.120,0,"depot")

@asynccontextmanager
async def lifespan(app):
    global clf
    import os
    os.makedirs(os.path.dirname(RESNET_PATH), exist_ok=True)
    if not os.path.exists(RESNET_PATH):
        logger.info("Initializing base model weights...")
        import torch
        from models.classifier import ResNet50WasteClassifier
        model = ResNet50WasteClassifier(num_classes=5, pretrained=True)
        torch.save(model.state_dict(), RESNET_PATH)
        logger.info(f"Base weights saved to {RESNET_PATH}")
    
    from models.classifier import WasteClassifier
    clf=WasteClassifier(resnet_weights=RESNET_PATH)
    logger.info("Classifier ready.")
    
    from sensors.sensors import SensorSimulator
    def _cache(t,p): _sensor_cache[t]=p
    sim=SensorSimulator(_cache,5.0); sim.start()
    FineTuneScheduler(FineTuner(store,RESNET_PATH,OUTPUT_PATH),86400).start()
    yield
    sim.stop()

app=FastAPI(title="Smart Waste API",version="1.0.0",lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the data folder to serve flagged images for the dashboard retraining queue
os.makedirs("data/flagged", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

class ClassifyResponse(BaseModel):
    label:str; class_id:int; confidence:float; bbox:Optional[list]
    latency_ms:float; flagged:bool

class FeedbackRequest(BaseModel):
    img_hash:str; true_label:str

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard HTML not found</h1>", status_code=404)

@app.get("/feedback/queue")
async def get_feedback_queue():
    return store._load()


@app.post("/classify",response_model=ClassifyResponse)
async def classify(file:UploadFile=File(...),background_tasks:BackgroundTasks=None):
    if clf is None: raise HTTPException(503,"Classifier not loaded.")
    if not file.content_type.startswith("image/"): raise HTTPException(400,"Must be an image.")
    img_bytes=await file.read()
    result=await asyncio.to_thread(clf.predict,img_bytes)
    _metrics["total"]+=1; _metrics["latency_sum"]+=result.latency_ms
    if result.flagged:
        _metrics["flagged"]+=1
        if background_tasks: background_tasks.add_task(store.save_flagged,img_bytes,result.label,result.confidence)
    return ClassifyResponse(label=result.label,class_id=result.class_id,confidence=result.confidence,
                            bbox=result.bbox,latency_ms=result.latency_ms,flagged=result.flagged)

@app.get("/sensors/live")
async def sensors_live(): return {"streams":len(_sensor_cache),"data":_sensor_cache}

@app.post("/feedback/flag")
async def feedback_flag(req:FeedbackRequest):
    try: store.submit_label(req.img_hash,req.true_label); return {"status":"ok"}
    except(KeyError,AssertionError)as e: raise HTTPException(400,str(e))

@app.get("/route/optimize")
async def route_optimize(threshold:float=60.0):
    opt=RouteOptimizer(DEMO_BINS,DEPOT)
    return await asyncio.to_thread(opt.optimize,threshold)

@app.get("/metrics")
async def metrics():
    elapsed=time.time()-_metrics["start_time"]; total=_metrics["total"]or 1
    return {"uptime_sec":round(elapsed),"total_classified":_metrics["total"],
            "flagged_count":_metrics["flagged"],"flag_rate":round(_metrics["flagged"]/total,4),
            "avg_latency_ms":round(_metrics["latency_sum"]/total,2),
            "throughput_per_hr":round(_metrics["total"]/max(elapsed/3600,1e-6)),
            "sensor_streams":len(_sensor_cache),"feedback_queue":store.queue_size()}

@app.post("/train/trigger")
async def train_trigger(background_tasks:BackgroundTasks):
    tuner=FineTuner(store,RESNET_PATH,OUTPUT_PATH,min_samples=5)
    background_tasks.add_task(tuner.run)
    return {"job_id":f"job_{int(time.time())}","status":"queued"}

@app.get("/health")
async def health(): return {"status":"ok","classifier":clf is not None,"sensors":len(_sensor_cache)}

if __name__=="__main__": uvicorn.run("api.main:app",host="0.0.0.0",port=8000,workers=4)
