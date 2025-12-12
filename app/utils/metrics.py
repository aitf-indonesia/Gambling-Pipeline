import time
from typing import Dict, List
from datetime import datetime


class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.success_count = 0
        self.error_count = 0
        self.gambling_count = 0
        self.non_gambling_count = 0
        self.latencies: List[float] = []
        self.component_times = {
            "classifier": [],
            "ocr": [],
            "detector": [],
            "visualization": []
        }
        self.start_time = time.time()
        self.last_request_time = None
    
    def record_request(self, success: bool, status: str, performance: Dict[str, float]):
        self.total_requests += 1
        self.last_request_time = datetime.now().isoformat()
        
        if success:
            self.success_count += 1
            
            if status == "gambling":
                self.gambling_count += 1
            else:
                self.non_gambling_count += 1
            
            self.latencies.append(performance.get("total_ms", 0))
            self.component_times["classifier"].append(performance.get("classifier_ms", 0))
            self.component_times["ocr"].append(performance.get("ocr_ms", 0))
            self.component_times["detector"].append(performance.get("detector_ms", 0))
            self.component_times["visualization"].append(performance.get("visualization_ms", 0))
        else:
            self.error_count += 1
    
    def get_stats(self) -> Dict:
        uptime = time.time() - self.start_time
        
        latency_stats = {}
        if self.latencies:
            latency_stats = {
                "avg_ms": round(sum(self.latencies) / len(self.latencies), 2),
                "min_ms": round(min(self.latencies), 2),
                "max_ms": round(max(self.latencies), 2)
            }
        
        component_avg = {}
        for component, times in self.component_times.items():
            if times:
                component_avg[f"{component}_ms"] = round(sum(times) / len(times), 2)
        
        success_rate = 0
        if self.total_requests > 0:
            success_rate = round((self.success_count / self.total_requests) * 100, 2)
        
        return {
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "gambling_detected": self.gambling_count,
            "non_gambling_detected": self.non_gambling_count,
            "latency": latency_stats,
            "component_avg": component_avg,
            "uptime_seconds": round(uptime, 2),
            "last_request": self.last_request_time
        }
