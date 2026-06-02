"""OCR backends for scanned PDF pages (Arabic + English).

Three interchangeable backends implement a single :class:`OCRBackend` contract
so the ingestion pipeline can transcribe a rendered page image to text without
knowing which engine is in use:

* :class:`QwenVLOCR`     — vision-language model (Qwen2.5-VL) prompted to
  transcribe all text in reading order; strongest on mixed AR/EN layouts.
* :class:`SuryaOCR`      — Surya detection + recognition for ``["ar", "en"]``.
* :class:`TesseractOCR`  — classic Tesseract via ``pytesseract``.

Heavy dependencies (transformers, surya, pytesseract, torch) are imported
**lazily** inside methods so this module stays importable on a CPU-only box and
passes ``py_compile`` without any of them installed. Models are loaded once and
cached on the instance.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from config import settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from PIL.Image import Image

logger = get_logger(__name__)


# The transcription instruction sent to the vision-language backend. Kept
# explicit so the model returns raw text only — no translation, no commentary.
_QWEN_INSTRUCTION = (
    "Transcribe ALL text visible in this image exactly as written, preserving "
    "the natural reading order. The page may contain Arabic and/or English. "
    "Keep Arabic text in Arabic and English text in English; do not translate, "
    "summarize, or add any commentary, labels, or explanations. Output only the "
    "transcribed text. If the page contains no readable text, output nothing."
)


class OCRBackend(abc.ABC):
    """Abstract base for an OCR engine that turns a page image into text."""

    name: str = "base"

    @abc.abstractmethod
    def ocr_image(self, image: "Image") -> str:
        """Return the text content of ``image`` (a ``PIL.Image.Image``)."""
        raise NotImplementedError


class TesseractOCR(OCRBackend):
    """Tesseract OCR via ``pytesseract``.

    Uses ``settings.ocr_lang`` (e.g. ``"ara+eng"``) so both scripts are
    recognized. Requires the Tesseract binary and the matching language data
    packs to be installed on the host.
    """

    name = "tesseract"

    def __init__(self, lang: str = settings.ocr_lang) -> None:
        self.lang = lang

    def ocr_image(self, image: "Image") -> str:
        import pytesseract  # lazy: heavy / optional dependency

        try:
            text = pytesseract.image_to_string(image, lang=self.lang)
        except pytesseract.TesseractError as exc:
            logger.error("Tesseract OCR failed (lang=%s): %s", self.lang, exc)
            raise
        return text.strip()


class QwenVLOCR(OCRBackend):
    """Qwen2.5-VL vision-language OCR.

    Loads the model + processor once (cached on the instance) and prompts the
    model to transcribe every glyph in reading order. Handles complex bilingual
    layouts better than line-based OCR engines.
    """

    name = "qwen"

    def __init__(
        self,
        model_name: str = settings.qwen_ocr_model,
        device: str = settings.qwen_ocr_device,
        max_new_tokens: int = settings.qwen_ocr_max_new_tokens,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        """Lazily instantiate the model + processor on first use."""
        if self._model is not None and self._processor is not None:
            return

        import torch  # lazy
        from transformers import AutoProcessor  # lazy

        logger.info("Loading Qwen2.5-VL OCR model %s on %s", self.model_name, self.device)

        # Prefer the dedicated class when present, else the generic
        # image-text-to-text auto class (transformers naming varies by version).
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _ModelClass
        except ImportError:  # pragma: no cover - depends on transformers version
            from transformers import AutoModelForImageTextToText as _ModelClass

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        device_map = "auto" if self.device.startswith("cuda") else None

        model = _ModelClass.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            model = model.to(self.device)
        model.eval()

        self._model = model
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def ocr_image(self, image: "Image") -> str:
        import torch  # lazy
        from qwen_vl_utils import process_vision_info  # lazy

        self._ensure_loaded()
        model = self._model
        processor = self._processor

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _QWEN_INSTRUCTION},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Strip the prompt tokens, keeping only the newly generated answer.
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()


class SuryaOCR(OCRBackend):
    """Surya OCR (detection + recognition) for Arabic and English.

    Loads the recognition and detection predictors once and caches them. Joins
    the recognized text lines (in detected reading order) into a single string.
    """

    name = "surya"

    LANGS = ["ar", "en"]

    def __init__(self) -> None:
        self._recognition: Any = None
        self._detection: Any = None
        self._api: str | None = None  # "predictor" (new) | "function" (legacy)

    def _ensure_loaded(self) -> None:
        if self._recognition is not None and self._detection is not None:
            return

        logger.info("Loading Surya OCR predictors (langs=%s)", self.LANGS)

        # Newer Surya exposes predictor classes; older releases expose
        # ``run_ocr`` plus model/processor loaders. Support both.
        try:
            from surya.recognition import RecognitionPredictor  # type: ignore
            from surya.detection import DetectionPredictor  # type: ignore

            self._recognition = RecognitionPredictor()
            self._detection = DetectionPredictor()
            self._api = "predictor"
            return
        except ImportError:
            pass

        # Legacy functional API.
        from surya.ocr import run_ocr  # type: ignore
        from surya.model.detection.model import (  # type: ignore
            load_model as load_det_model,
            load_processor as load_det_processor,
        )
        from surya.model.recognition.model import load_model as load_rec_model  # type: ignore
        from surya.model.recognition.processor import (  # type: ignore
            load_processor as load_rec_processor,
        )

        self._run_ocr = run_ocr
        self._det_model = load_det_model()
        self._det_processor = load_det_processor()
        self._rec_model = load_rec_model()
        self._rec_processor = load_rec_processor()
        self._recognition = self._rec_model
        self._detection = self._det_model
        self._api = "function"

    def ocr_image(self, image: "Image") -> str:
        self._ensure_loaded()

        if self._api == "predictor":
            predictions = self._recognition(
                [image], [self.LANGS], det_predictor=self._detection
            )
        else:  # legacy functional API
            predictions = self._run_ocr(
                [image],
                [self.LANGS],
                self._det_model,
                self._det_processor,
                self._rec_model,
                self._rec_processor,
            )

        if not predictions:
            return ""

        result = predictions[0]
        text_lines = getattr(result, "text_lines", None)
        if text_lines is None:
            return ""

        lines = [getattr(line, "text", "") for line in text_lines]
        return "\n".join(line for line in (l.strip() for l in lines) if line)


# Map backend name -> constructor. Extend here to register new engines.
_BACKENDS: dict[str, type[OCRBackend]] = {
    "qwen": QwenVLOCR,
    "surya": SuryaOCR,
    "tesseract": TesseractOCR,
}


def get_ocr_backend(name: str = settings.ocr_backend) -> OCRBackend:
    """Construct the OCR backend selected by ``name`` (``qwen|surya|tesseract``).

    The heavy model is not loaded until the backend's :meth:`ocr_image` is
    first called, so this factory is cheap to call.
    """
    key = (name or "").strip().lower()
    backend_cls = _BACKENDS.get(key)
    if backend_cls is None:
        valid = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown OCR backend {name!r}; choose one of: {valid}")
    logger.debug("Selected OCR backend: %s", key)
    return backend_cls()
