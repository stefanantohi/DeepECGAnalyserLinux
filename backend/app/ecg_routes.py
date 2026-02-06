"""
ECG Analysis API Routes.

Provides two modes:
1. Single Model Analysis (POST /ecg/predict) - Fast, targeted analysis
2. Global Screening (POST /ecg/screen) - Comprehensive multi-model analysis
"""
import asyncio
import logging
import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from .settings import settings, ECG_MODELS, DEFAULT_SCREENING_MODELS
from .ai_engine_client import analyze_file, check_ai_engine_health

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ecg", tags=["ECG Analysis"])


# Request/Response Models
class PredictRequest(BaseModel):
    """Single model prediction request."""
    model_id: str


class PredictResponse(BaseModel):
    """Single model prediction response."""
    success: bool
    model_id: str
    model_name: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float
    request_id: str


class ScreeningRequest(BaseModel):
    """Screening request with optional model selection."""
    models: Optional[List[str]] = None  # None = all default models
    policy: str = "all"  # "all", "cascaded", "parallel"


class ModelResult(BaseModel):
    """Individual model result in screening."""
    model_id: str
    model_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float


class ScreeningSummary(BaseModel):
    """Summary of screening results."""
    flagged: List[str] = []
    warnings: List[str] = []
    recommended_next_steps: List[str] = []


class ScreeningResponse(BaseModel):
    """Complete screening response."""
    success: bool
    summary: ScreeningSummary
    results: Dict[str, ModelResult]
    meta: Dict[str, Any]
    request_id: str


class ModelsListResponse(BaseModel):
    """List of available models."""
    models: Dict[str, Dict[str, Any]]
    default_screening: List[str]
    categories: List[str]


# Endpoints
@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all available ECG analysis models."""
    categories = list(set(m["category"] for m in ECG_MODELS.values()))
    return ModelsListResponse(
        models=ECG_MODELS,
        default_screening=DEFAULT_SCREENING_MODELS,
        categories=sorted(categories)
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_single(
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    Run a single model prediction on ECG data.

    Fast, targeted analysis for specific clinical questions.

    Args:
        file: ECG data file (CSV or Parquet)
        model_id: ID of the model to use (e.g., "ecg_77labels", "lvef_40")

    Returns:
        Prediction result from the specified model
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Validate model
    if model_id not in ECG_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}. Available: {list(ECG_MODELS.keys())}"
        )

    model_info = ECG_MODELS[model_id]

    # Validate file extension
    filename = file.filename or "unknown"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in settings.ALLOWED_DATA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_DATA_EXTENSIONS}"
        )

    try:
        # Read file content
        content = await file.read()

        # Call AI Engine with specific model
        result = await analyze_file(
            file_content=content,
            filename=filename,
            model_id=model_id
        )

        processing_time = (time.time() - start_time) * 1000

        return PredictResponse(
            success=result.get("success", False),
            model_id=model_id,
            model_name=model_info["name"],
            result=result.get("outputs"),
            error=result.get("error"),
            processing_time_ms=processing_time,
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"Prediction error for model {model_id}: {e}")
        processing_time = (time.time() - start_time) * 1000
        return PredictResponse(
            success=False,
            model_id=model_id,
            model_name=model_info["name"],
            error=str(e),
            processing_time_ms=processing_time,
            request_id=request_id
        )


@router.post("/screen", response_model=ScreeningResponse)
async def screen_ecg(
    file: UploadFile = File(...),
    models: Optional[str] = Form(None),  # Comma-separated list or None for defaults
    policy: str = Form("all")  # "all", "cascaded", "parallel"
):
    """
    Run comprehensive ECG screening with multiple models.

    Three policies available:
    - "all": Run all specified models
    - "cascaded": Smart cascade - run models based on triggers from previous results
    - "parallel": Run all models in parallel (fastest, but uses more resources)

    Args:
        file: ECG data file (CSV or Parquet)
        models: Comma-separated model IDs, or None for default screening set
        policy: Execution policy ("all", "cascaded", "parallel")

    Returns:
        Aggregated results from all models with clinical summary
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Parse models list
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        # Validate all models
        invalid = [m for m in model_list if m not in ECG_MODELS]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown models: {invalid}. Available: {list(ECG_MODELS.keys())}"
            )
    else:
        model_list = DEFAULT_SCREENING_MODELS.copy()

    # Validate file
    filename = file.filename or "unknown"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in settings.ALLOWED_DATA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_DATA_EXTENSIONS}"
        )

    try:
        # Read file content once
        content = await file.read()

        results: Dict[str, ModelResult] = {}

        if policy == "parallel":
            # Run all models in parallel
            results = await _run_models_parallel(content, filename, model_list)
        elif policy == "cascaded":
            # Run models with smart cascading
            results = await _run_models_cascaded(content, filename, model_list)
        else:
            # Run all models sequentially ("all" policy)
            results = await _run_models_sequential(content, filename, model_list)

        # Generate clinical summary
        summary = _generate_summary(results)

        total_time = (time.time() - start_time) * 1000

        return ScreeningResponse(
            success=True,
            summary=summary,
            results=results,
            meta={
                "policy": policy,
                "models_requested": model_list,
                "models_executed": list(results.keys()),
                "total_time_ms": total_time,
                "filename": filename
            },
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"Screening error: {e}")
        total_time = (time.time() - start_time) * 1000
        return ScreeningResponse(
            success=False,
            summary=ScreeningSummary(warnings=[str(e)]),
            results={},
            meta={
                "policy": policy,
                "error": str(e),
                "total_time_ms": total_time
            },
            request_id=request_id
        )


async def _run_models_sequential(
    content: bytes,
    filename: str,
    model_list: List[str]
) -> Dict[str, ModelResult]:
    """Run models one by one."""
    results = {}

    for model_id in model_list:
        model_info = ECG_MODELS[model_id]
        start = time.time()

        try:
            result = await analyze_file(
                file_content=content,
                filename=filename,
                model_id=model_id
            )

            results[model_id] = ModelResult(
                model_id=model_id,
                model_name=model_info["name"],
                success=result.get("success", False),
                result=result.get("outputs"),
                error=result.get("error"),
                processing_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            results[model_id] = ModelResult(
                model_id=model_id,
                model_name=model_info["name"],
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start) * 1000
            )

    return results


async def _run_models_parallel(
    content: bytes,
    filename: str,
    model_list: List[str]
) -> Dict[str, ModelResult]:
    """Run all models in parallel."""

    async def run_single(model_id: str) -> tuple[str, ModelResult]:
        model_info = ECG_MODELS[model_id]
        start = time.time()

        try:
            result = await analyze_file(
                file_content=content,
                filename=filename,
                model_id=model_id
            )

            return model_id, ModelResult(
                model_id=model_id,
                model_name=model_info["name"],
                success=result.get("success", False),
                result=result.get("outputs"),
                error=result.get("error"),
                processing_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return model_id, ModelResult(
                model_id=model_id,
                model_name=model_info["name"],
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start) * 1000
            )

    # Run all in parallel
    tasks = [run_single(m) for m in model_list]
    task_results = await asyncio.gather(*tasks)

    return {model_id: result for model_id, result in task_results}


async def _run_models_cascaded(
    content: bytes,
    filename: str,
    model_list: List[str]
) -> Dict[str, ModelResult]:
    """
    Run models with smart cascading based on triggers.

    Priority-based execution:
    1. Run priority 1 models first (interpretation)
    2. Based on results, determine which priority 2+ models to run
    """
    results = {}

    # Sort by priority
    sorted_models = sorted(
        model_list,
        key=lambda m: ECG_MODELS[m].get("priority", 99)
    )

    for model_id in sorted_models:
        model_info = ECG_MODELS[model_id]

        # Check if this model has triggers and if they're satisfied
        triggers = model_info.get("triggers", [])
        if triggers:
            # Check if trigger model was run and had relevant findings
            should_run = False
            for trigger_model in triggers:
                if trigger_model in results:
                    trigger_result = results[trigger_model]
                    if trigger_result.success and _should_trigger(trigger_result, model_id):
                        should_run = True
                        break

            if not should_run:
                logger.info(f"Skipping {model_id} - trigger conditions not met")
                continue

        # Run this model
        start = time.time()
        try:
            result = await analyze_file(
                file_content=content,
                filename=filename,
                model_id=model_id
            )

            results[model_id] = ModelResult(
                model_id=model_id,
                model_name=model_info["name"],
                success=result.get("success", False),
                result=result.get("outputs"),
                error=result.get("error"),
                processing_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            results[model_id] = ModelResult(
                model_id=model_id,
                model_name=model_info["name"],
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start) * 1000
            )

    return results


def _should_trigger(trigger_result: ModelResult, target_model: str) -> bool:
    """
    Determine if a target model should be triggered based on results.

    This is where clinical logic goes - customize based on your needs.
    """
    if not trigger_result.result:
        return False

    # Example trigger logic (customize as needed)
    result = trigger_result.result

    # LQTS detection triggered by QT prolongation in interpretation
    if target_model == "lqts_detect":
        labels = result.get("labels", [])
        scores = result.get("scores", {})
        # Look for QT-related findings
        qt_keywords = ["qt prolongation", "long qt", "qt interval"]
        for label in labels:
            if any(kw in label.lower() for kw in qt_keywords):
                return True
        return False

    # LQTS genotype triggered by positive LQTS detection
    if target_model == "lqts_genotype":
        prob = result.get("probability", result.get("prob", 0))
        return prob > 0.5

    # LVEF triggered by heart failure indicators
    if target_model in ["lvef_40", "lvef_50"]:
        labels = result.get("labels", [])
        hf_keywords = ["heart failure", "cardiomyopathy", "reduced ef", "lvh"]
        for label in labels:
            if any(kw in label.lower() for kw in hf_keywords):
                return True
        return False

    # Default: always trigger
    return True


def _generate_summary(results: Dict[str, ModelResult]) -> ScreeningSummary:
    """
    Generate clinical summary from screening results.

    Aggregates findings and provides recommendations.
    """
    flagged = []
    warnings = []
    recommendations = []

    for model_id, result in results.items():
        if not result.success:
            warnings.append(f"{result.model_name}: Analysis failed")
            continue

        if not result.result:
            continue

        data = result.result

        # Check for positive/high-risk findings based on model
        if model_id == "lvef_40":
            prob = data.get("probability", data.get("prob", 0))
            if prob > 0.5:
                flagged.append("LVEF_low_probability")
                recommendations.append("Confirm with echocardiography")

        elif model_id == "lvef_50":
            prob = data.get("probability", data.get("prob", 0))
            if prob > 0.5:
                flagged.append("LVEF_moderate_reduction")
                recommendations.append("Consider cardiac imaging")

        elif model_id == "af_5y":
            risk = data.get("risk", data.get("probability", 0))
            if risk > 0.2:
                flagged.append("AF_risk_elevated")
                recommendations.append("Consider Holter monitoring if symptomatic")

        elif model_id == "lqts_detect":
            prob = data.get("probability", data.get("prob", 0))
            if prob > 0.5:
                flagged.append("LQTS_suspected")
                recommendations.append("Genetic testing recommended")
                recommendations.append("Avoid QT-prolonging medications")

        elif model_id == "hcm_detect":
            prob = data.get("probability", data.get("prob", 0))
            if prob > 0.5:
                flagged.append("HCM_suspected")
                recommendations.append("Cardiac MRI recommended")

        elif model_id == "mortality_risk":
            risk = data.get("risk", data.get("probability", 0))
            if risk > 0.3:
                flagged.append("Elevated_mortality_risk")
                recommendations.append("Comprehensive cardiac evaluation")

        elif model_id == "ecg_77labels":
            # Check for critical labels
            labels = data.get("top_labels", data.get("labels", []))
            critical_keywords = ["infarction", "ischemia", "block", "arrhythmia"]
            for label_info in labels[:5]:  # Check top 5
                label = label_info if isinstance(label_info, str) else label_info.get("label", "")
                if any(kw in label.lower() for kw in critical_keywords):
                    flagged.append(f"ECG_finding: {label}")

    # Deduplicate recommendations
    recommendations = list(dict.fromkeys(recommendations))

    return ScreeningSummary(
        flagged=flagged,
        warnings=warnings,
        recommended_next_steps=recommendations
    )
