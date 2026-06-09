"""Adaptive ML Feedback Loop — flag, store, fine-tune."""
import json,os,shutil,logging,hashlib,time,threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass,asdict
from typing import Optional
logger=logging.getLogger(__name__)
FLAGGED_DIR=Path("data/flagged"); TRAINING_DIR=Path("data/training")
METADATA_FILE=Path("data/feedback_metadata.jsonl")
CLASSES=["Plastic","Paper","Glass","Metal","Organic"]

@dataclass
class FlaggedItem:
    img_hash:str; predicted_label:str; confidence:float
    true_label:str; timestamp:str; status:str

class FeedbackStore:
    def __init__(self):
        for d in [FLAGGED_DIR,TRAINING_DIR]: d.mkdir(parents=True,exist_ok=True)
        METADATA_FILE.parent.mkdir(parents=True,exist_ok=True)

    def save_flagged(self,image_bytes,predicted,confidence):
        h=hashlib.sha256(image_bytes).hexdigest()[:12]
        (FLAGGED_DIR/f"{h}.jpg").write_bytes(image_bytes)
        item=FlaggedItem(h,predicted,round(confidence,4),"",
                         datetime.utcnow().isoformat()+"Z","pending")
        with METADATA_FILE.open("a") as f: f.write(json.dumps(asdict(item))+"\n")
        return h

    def submit_label(self,img_hash,true_label):
        assert true_label in CLASSES
        records=self._load()
        for r in records:
            if r["img_hash"]==img_hash: r["true_label"]=true_label; r["status"]="reviewed"
        self._write(records)

    def get_training_batch(self,min_count=30):
        ready=[r for r in self._load() if r["status"]=="reviewed" and r["true_label"]]
        return ready if len(ready)>=min_count else []

    def mark_used(self,hashes):
        records=self._load()
        for r in records:
            if r["img_hash"] in hashes:
                r["status"]="used_in_training"
                src=FLAGGED_DIR/f"{r['img_hash']}.jpg"
                dst=TRAINING_DIR/f"{r['true_label']}_{r['img_hash']}.jpg"
                if src.exists(): shutil.move(str(src),str(dst))
        self._write(records)

    def queue_size(self):
        rec=self._load()
        return {s:sum(1 for r in rec if r["status"]==s) for s in ["pending","reviewed","used_in_training"]}

    def _load(self):
        if not METADATA_FILE.exists(): return []
        with METADATA_FILE.open() as f: return [json.loads(l) for l in f if l.strip()]

    def _write(self,records):
        with METADATA_FILE.open("w") as f:
            for r in records: f.write(json.dumps(r)+"\n")

class FineTuner:
    def __init__(self,store,model_path,output_path,min_samples=30):
        self.store=store; self.model_path=model_path
        self.output_path=output_path; self.min_samples=min_samples

    def run(self):
        batch=self.store.get_training_batch(self.min_samples)
        if not batch: return {"status":"skipped","reason":"insufficient_samples"}
        import torch,torch.nn as nn
        from torch.utils.data import Dataset,DataLoader
        from torchvision import transforms
        from models.classifier import ResNet50WasteClassifier
        from PIL import Image

        class FDS(Dataset):
            def __init__(self,records,tfm): self.records=records; self.tfm=tfm
            def __len__(self): return len(self.records)
            def __getitem__(self,idx):
                r=self.records[idx]
                p=FLAGGED_DIR/f"{r['img_hash']}.jpg"
                if not p.exists(): p=TRAINING_DIR/f"{r['true_label']}_{r['img_hash']}.jpg"
                return self.tfm(Image.open(p).convert("RGB")), CLASSES.index(r["true_label"])

        tfm=transforms.Compose([transforms.RandomResizedCrop(224,(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2),transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=ResNet50WasteClassifier(num_classes=len(CLASSES))
        model.load_state_dict(torch.load(self.model_path,map_location=device))
        model.to(device)
        for p in model.backbone.parameters(): p.requires_grad=False
        for p in model.backbone.fc.parameters(): p.requires_grad=True
        loader=DataLoader(FDS(batch,tfm),16,shuffle=True,num_workers=2)
        optim=torch.optim.AdamW(model.backbone.fc.parameters(),lr=1e-4)
        loss_fn=nn.CrossEntropyLoss(); total=0.0; epochs=5
        model.train()
        for _ in range(epochs):
            for imgs,labels in loader:
                imgs,labels=imgs.to(device),labels.to(device)
                optim.zero_grad(); loss=loss_fn(model(imgs),labels)
                loss.backward(); optim.step(); total+=loss.item()
        model.eval(); torch.save(model.state_dict(),self.output_path)
        self.store.mark_used([r["img_hash"] for r in batch])
        return {"status":"completed","samples":len(batch),"epochs":epochs,
                "avg_loss":round(total/(epochs*max(len(loader),1)),4)}

class FineTuneScheduler:
    def __init__(self,tuner,interval_sec=86400):
        self.tuner=tuner; self.interval=interval_sec
    def start(self):
        def _loop():
            while True:
                time.sleep(self.interval)
                try: self.tuner.run()
                except Exception as e: logger.error(f"Fine-tune error: {e}")
        threading.Thread(target=_loop,daemon=True).start()
