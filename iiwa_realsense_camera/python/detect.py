from __future__ import annotations
from ament_index_python.packages import get_package_share_path
from typing import Any, Literal, Union
from ultralytics import YOLO
from pathlib import Path

import importlib.resources as pkg
import numpy as np

TaskT = Literal["detect", "segment", "classify", "pose", "obb"]
FamilyT = Literal["yolo8", "yolo11"]

_model: Union[YOLO, None] = None
_task: TaskT = "pose"

_pkg_root = get_package_share_path("iiwa_realsense_camera")
MODELS_DIR = _pkg_root / "models"


def _slice(arr: np.ndarray, n: int | None):
    return arr if n is None or arr.shape[0] <= n else arr[:n]


def _family_from_model(model_name: str) -> FamilyT | None:
    if model_name.startswith("yolo8"):
        return "yolo8"
    if model_name.startswith("yolo11"):
        return "yolo11"
    return None


def _resolve_model_path(task: TaskT,
                        model_name: Union[str, None],
                        model_format: str) -> Path:

    family: FamilyT = "yolo8"

    if model_name:
        if "/" in model_name:
            return MODELS_DIR / model_name
        fam = _family_from_model(model_name)
        if fam:
            family = fam
        fname = f"{model_name}.{model_format}" if "." not in model_name else model_name
        return MODELS_DIR / family / task / fname

    task_dir = MODELS_DIR / family / task
    for p in task_dir.glob("*.pt"):
        return p
    raise FileNotFoundError(f"No model files in {task_dir}")


def _load(task: TaskT, model_path: Path):
    global _model, _task
    if _model is None or _task != task or str(model_path) not in str(_model):
        _task  = task
        _model = YOLO(str(model_path))
        print(f"[detect] loaded {_task} model from {model_path}")


def load_model(task: TaskT = "pose", 
               model_name: Union[str, None] = None, 
               model_format: str="pt") -> dict:
        
    task = task.lower()
    try:
        path = _resolve_model_path(task, model_name, model_format)
    except Exception as e:
        return {"status": False, "message": str(e)}

    _load(task, path)
    return {"status": True, "message": f"Model for '{task}' loaded: {path.name}"}
    
    
def detect(img: np.ndarray,
           *,
           conf: float = 0.25,
           task: TaskT | None = None,
           max_objects: int | None = None,
           device: str = "cpu") -> dict[str, Any]:
    
    global _task
    task = (task or _task).lower()
    if _model is None or _task != task:
        load_model(task)

    n_limit = max_objects if max_objects is not None else None
    res = _model(img, verbose=False, conf=conf, device=device)[0]
    
    out: dict[str, Any] = {
        "task" : task,
        "boxes" : np.empty((0, 4), dtype=np.float32),
        "scores" : np.empty((0,), dtype=np.float32),
        "class_id" : np.empty((0,), dtype=np.int32),
        "keypoints" : np.empty((0, 17, 2), dtype=np.float32),
        "masks" : np.empty((0,), dtype=bool),
        "obb" : np.empty((0, 5), dtype=np.float32),
    }
    
    # ---------- task-specific containers ---------------------------------
    if task == "obb":
        confs = res.obb.conf.cpu().numpy()
        clses = res.obb.cls.cpu().numpy().astype(np.int32)
        out["obb"] = _slice(res.obb.xywhr.cpu().numpy(), n_limit)
    elif task == "classify":
        tensor = res.probs.data if hasattr(res.probs, "data") else res.probs
        probs  = tensor.cpu().numpy()            # ndarray(float32)
        cls_id = int(np.argmax(probs))
        score  = float(probs[cls_id])

        out["scores"]   = np.array([score],  dtype=np.float32)
        out["class_id"] = np.array([cls_id], dtype=np.int32)
        return out
    else:
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(np.int32)
        out["boxes"] = _slice(res.boxes.xyxy.cpu().numpy(), n_limit)

    out["scores"]   = _slice(confs, n_limit)
    out["class_id"] = _slice(clses, n_limit)

    if task in {"detect", "pose", "segment"}:
        out["boxes"] = _slice(res.boxes.xyxy.cpu().numpy(), n_limit)

    if task == "pose" and res.keypoints and res.keypoints.xy is not None:
        out["keypoints"] = _slice(res.keypoints.xy.cpu().numpy().astype("float32"), n_limit)

    if task == "segment" and res.masks is not None:
        out["masks"] = _slice(res.masks.data.cpu().numpy().astype(bool), n_limit)

    if task == "obb":
        out["obb"] = _slice(res.obb.xywhr.cpu().numpy(), n_limit)
        
    

    return out
