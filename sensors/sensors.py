"""IoT Sensor Simulation — 12 MQTT streams."""
import json, math, random, time, threading, logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

BROKER_HOST, BROKER_PORT = "localhost", 1883
BASE_TOPIC = "waste"

@dataclass
class SensorReading:
    sensor_id: str; zone: str; bin_id: str; sensor_type: str
    value: float; unit: str; timestamp: str
    latitude: float = 0.0; longitude: float = 0.0
    def to_json(self): return json.dumps(asdict(self))
    @property
    def topic(self): return f"{BASE_TOPIC}/{self.zone}/{self.bin_id}/{self.sensor_type}"

class SensorSimulator:
    SENSOR_CONFIG = [
        ("S01","zone_A","bin_01","ultrasonic",72.0,"percent",51.5074,-0.1278),
        ("S02","zone_A","bin_01","weight",38.5,"kg",51.5074,-0.1278),
        ("S03","zone_A","bin_02","ultrasonic",41.0,"percent",51.5080,-0.1290),
        ("S04","zone_A","bin_02","temperature",22.5,"celsius",51.5080,-0.1290),
        ("S05","zone_B","bin_03","ultrasonic",88.0,"percent",51.5100,-0.1300),
        ("S06","zone_B","bin_03","weight",51.2,"kg",51.5100,-0.1300),
        ("S07","zone_B","bin_03","humidity",58.0,"percent",51.5100,-0.1300),
        ("S08","zone_B","bin_04","gas_voc",120.0,"ppm",51.5120,-0.1310),
        ("S09","zone_C","bin_05","ultrasonic",33.0,"percent",51.5140,-0.1320),
        ("S10","zone_C","bin_05","weight",18.9,"kg",51.5140,-0.1320),
        ("S11","zone_C","bin_06","ultrasonic",65.0,"percent",51.5160,-0.1330),
        ("S12","zone_C","bin_06","temperature",24.1,"celsius",51.5160,-0.1330),
    ]
    def __init__(self, publish_cb=None, interval_sec=5.0):
        self.publish_cb = publish_cb or (lambda t,p: print(f"[MQTT] {t}: {p}"))
        self.interval = interval_sec
        self._state = {c[0]: c[4] for c in self.SENSOR_CONFIG}
        self._running = False

    def _drift(self, sid, cfg):
        stype = cfg[3]; val = self._state[sid]; t = time.time()
        if stype == "ultrasonic":
            val += random.gauss(0.2,0.8)
            if val > 95: val = random.uniform(5,15)
        elif stype == "weight": val = max(0,min(60, val+random.gauss(0.15,0.4)))
        elif stype == "temperature":
            hr = (t%86400)/86400*2*math.pi; val = 20+8*math.sin(hr-math.pi/2)+random.gauss(0,0.3)
        elif stype == "humidity": val = max(30,min(95, val+random.gauss(0,0.5)))
        elif stype == "gas_voc": val = max(10,min(500, val+random.gauss(0,5)))
        val = max(0,val); self._state[sid] = val; return round(val,2)

    def tick(self):
        for cfg in self.SENSOR_CONFIG:
            sid,zone,bin_id,stype,_,unit,lat,lon = cfg
            val = self._drift(sid, cfg)
            r = SensorReading(sid,zone,bin_id,stype,val,unit,
                              datetime.utcnow().isoformat()+"Z",
                              lat+random.gauss(0,1e-5), lon+random.gauss(0,1e-5))
            self.publish_cb(r.topic, r.to_json())

    def start(self):
        self._running = True
        def _loop():
            while self._running: self.tick(); time.sleep(self.interval)
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self): self._running = False

def build_mqtt_publisher(host=BROKER_HOST, port=BROKER_PORT):
    if not MQTT_AVAILABLE: return lambda t,p: print(f"[STUB] {t}: {p}")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(host, port, 60); client.loop_start()
    return lambda t,p: client.publish(t, p, qos=1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = SensorSimulator(build_mqtt_publisher(), 5.0)
    sim.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: sim.stop()
