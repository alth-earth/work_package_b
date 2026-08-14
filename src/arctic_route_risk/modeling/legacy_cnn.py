"""CPU-only, one-step adapter for the converted legacy CNN asset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from arctic_route_risk.errors import RiskPipelineError
from arctic_route_risk.modeling.contracts import (
    MODEL_ARTIFACT_INVALID,
    MODEL_INPUT_INCOMPATIBLE,
    MODEL_OUTPUT_INVALID,
    MODEL_RUNTIME_UNAVAILABLE,
    ModelArtifactManifest,
    RiskModelInput,
    RiskModelOutput,
    model_error,
)


class LegacyCnnOneStepBackend:
    """Read only ``model.safetensors`` and return one un-timed risk grid."""

    backend_id = "legacy_cnn_one_step"
    required_variables = ("comprehensive_risk",)

    def __init__(self, artifact_dir: str | Path = "models/legacy_cnn_one_step_v1") -> None:
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "manifest.json"
        tensor_path = self.artifact_dir / "model.safetensors"
        try:
            manifest = ModelArtifactManifest.from_document(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except RiskPipelineError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise model_error(
                MODEL_ARTIFACT_INVALID, f"cannot load manifest: {manifest_path}"
            ) from exc
        except Exception as exc:
            raise model_error(
                MODEL_ARTIFACT_INVALID, f"cannot load manifest: {manifest_path}"
            ) from exc
        if manifest.backend_id != self.backend_id or manifest.time_step_status != "unknown":
            raise model_error(
                MODEL_ARTIFACT_INVALID, "manifest is not a legacy unknown-cadence asset"
            )
        if not tensor_path.is_file():
            raise model_error(MODEL_ARTIFACT_INVALID, f"missing safetensors: {tensor_path}")
        digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        if digest != manifest.safetensors_sha256:
            raise model_error(MODEL_ARTIFACT_INVALID, "safetensors hash does not match manifest")
        try:
            import torch
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - optional environment
            raise model_error(MODEL_RUNTIME_UNAVAILABLE, "install the model-cpu extra") from exc
        try:
            state = load_file(str(tensor_path), device="cpu")

            class _LegacyCnn(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Conv2d(1, 32, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 32, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 1, 1),
                        torch.nn.Sigmoid(),
                    )

                def forward(self, values):
                    return self.net(values)

            model = _LegacyCnn()
            model.load_state_dict(state, strict=True)
            model.to(device="cpu")
            model.eval()
            torch.set_num_threads(1)
        except Exception as exc:
            raise model_error(
                MODEL_ARTIFACT_INVALID, "safetensors does not match the CNN architecture"
            ) from exc
        self.manifest = manifest
        self.model_version = manifest.model_version
        self._torch = torch
        self._model = model

    def infer(self, model_input: RiskModelInput) -> RiskModelOutput:
        if set(model_input.variables) != set(self.required_variables):
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE, "legacy CNN accepts exactly comprehensive_risk"
            )
        if (
            model_input.requested_valid_time is not None
            or model_input.time_step_status != "unknown"
        ):
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE,
                "legacy CNN cannot accept a requested or declared output cadence",
            )
        values = np.asarray(model_input.variables["comprehensive_risk"])
        if values.ndim != 2:
            raise model_error(MODEL_INPUT_INCOMPATIBLE, "comprehensive_risk must be 2-D")
        if not np.issubdtype(values.dtype, np.number):
            raise model_error(MODEL_INPUT_INCOMPATIBLE, "comprehensive_risk must be numeric")
        if values.shape != (model_input.latitude.size, model_input.longitude.size):
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE, "risk grid shape does not match coordinates"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE, "comprehensive_risk must be finite in [0, 1]"
            )
        for name, coordinate in (
            ("latitude", model_input.latitude),
            ("longitude", model_input.longitude),
        ):
            differences = np.diff(coordinate)
            if not np.allclose(
                differences,
                self.manifest.native_grid_step_degrees,
                rtol=0,
                atol=1e-8,
            ):
                raise model_error(
                    MODEL_INPUT_INCOMPATIBLE,
                    (
                        f"{name} spacing must be exactly "
                        f"{self.manifest.native_grid_step_degrees} degrees"
                    ),
                )
        try:
            tensor = (
                self._torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            with self._torch.inference_mode():
                prediction = self._model(tensor).squeeze(0).squeeze(0)
            output = prediction.detach().cpu().numpy().astype(np.float32, copy=True)
        except Exception as exc:
            raise model_error(MODEL_RUNTIME_UNAVAILABLE, "CPU CNN inference failed") from exc
        if (
            output.shape != values.shape
            or not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise model_error(MODEL_OUTPUT_INVALID, "CNN output is not finite and bounded")
        output.setflags(write=False)
        return RiskModelOutput(
            risk_score=output,
            predicted_valid_time=None,
            time_step_status="unknown",
            backend_id=self.backend_id,
            model_version=self.model_version,
            metadata={
                "artifact_id": self.manifest.artifact_id,
                "native_grid_step_degrees": self.manifest.native_grid_step_degrees,
                "runtime": "cpu",
                "formal_risk_frame": False,
            },
        )


__all__ = ["LegacyCnnOneStepBackend"]
