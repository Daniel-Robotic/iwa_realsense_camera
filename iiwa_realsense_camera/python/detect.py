from __future__ import annotations
from ament_index_python.packages import get_package_share_path
from typing import Any, Literal, Union
from ultralytics import YOLO
from pathlib import Path

import importlib.resources as pkg
import numpy as np
import torch

# Define the supported task types and model families
TaskT = Literal["detect", "segment", "classify", "pose", "obb"]
FamilyT = Literal["yolo8", "yolo11"]

# Global model instance and task type
_model: Union[YOLO, None] = None
_task: TaskT = "pose"

# Get the path to the models directory inside the ROS package
_pkg_root = get_package_share_path("iiwa_realsense_camera")
MODELS_DIR = _pkg_root / "models"


def _slice(arr: np.ndarray, n: int | None):
    """Return the first n elements of the array if n is specified and less than the array size."""
    return arr if n is None or arr.shape[0] <= n else arr[:n]


def _family_from_model(model_name: str) -> FamilyT | None:
    """Determine the model family based on the model name prefix."""
    if model_name.startswith("yolo8"):
        return "yolo8"
    if model_name.startswith("yolo11"):
        return "yolo11"
    return None


def _resolve_model_path(task: TaskT,
                        model_name: Union[str, None],
                        model_format: str) -> Path:

    """
    Determine the full path to the model file.
    If model_name is not given, fallback to the first available model for the task.
    """
    
    family: FamilyT = "yolo8"

    if model_name:
        if "/" in model_name:
            return MODELS_DIR / model_name
        fam = _family_from_model(model_name)
        if fam:
            family = fam
        fname = f"{model_name}.{model_format}" if "." not in model_name else model_name
        return MODELS_DIR / family / task / fname

    # If no model name is provided, use the first .pt model found in the task directory
    task_dir = MODELS_DIR / family / task
    for p in task_dir.glob("*.pt"):
        return p
    raise FileNotFoundError(f"No model files in {task_dir}")


def _load(task: TaskT, model_path: Path, device: str = "cuda"):
    """
    Load the model into memory. If the model is in .pt format and no .engine file exists,
    it will be converted to TensorRT.
    """
    
    global _model, _task

    if model_path.suffix == ".pt":
        engine_path = model_path.with_suffix(".engine")
        if not engine_path.exists():
            try:
                print(f"[convert] Converting {model_path.name} to TensorRT ({engine_path.name})...")
                model = YOLO(str(model_path), verbose=False)

                if device != "cpu":
                    torch.cuda.empty_cache()

                model.export(
                    format="engine",
                    device=device,
                    half=True,
                    workspace=2.0,
                    simplify=True,
                    verbose=False
                )
                
            except Exception as e:
                raise RuntimeError(f"Failed to convert {model_path} to TensorRT: {e}")
        
        model_path = engine_path

    if _model is None or _task != task or str(model_path) not in str(_model):
        _task = task
        _model = YOLO(str(model_path))
        print(f"[detect] loaded {_task} model from {model_path}")


def load_model(task: TaskT = "pose", 
               model_name: Union[str, None] = None, 
               model_format: str = "pt",
               device: str = "cuda") -> dict:
    """
    Load and optionally convert a model for the given task.
    Returns a dictionary with status and message.
    """
    
    task = task.lower()
    try:
        path = _resolve_model_path(task, model_name, model_format)
    except Exception as e:
        return {"status": False, "message": str(e)}

    try:
        _load(task, path, device=device)
    except Exception as e:
        return {"status": False, "message": f"Failed to load model: {e}"}

    return {"status": True, "message": f"Model for '{task}' loaded: {path.name}"}
    
    
def detect(img: np.ndarray,
           *,
           conf: float = 0.25,
           task: TaskT | None = None,
           max_objects: int | None = None,
           device: str = "cpu") -> dict[str, Any]:
    """
    Run inference on the image using the currently loaded model.
    Supports tasks: detect, segment, pose, classify, obb.
    Returns a dictionary with structured results for the given task.
    """
    
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
    
    if task == "obb":
        confs = res.obb.conf.cpu().numpy()
        clses = res.obb.cls.cpu().numpy().astype(np.int32)
        out["obb"] = _slice(res.obb.xywhr.cpu().numpy(), n_limit)
    elif task == "classify":
        tensor = res.probs.data if hasattr(res.probs, "data") else res.probs
        probs  = tensor.cpu().numpy()
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
