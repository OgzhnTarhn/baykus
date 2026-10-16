"""INT8 nicemleme testleri (Görev 6): ORT statik QDQ, kanal başına, isteğe bağlı FP32 baş."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnxruntime.quantization import CalibrationDataReader

from baykus.edge.quantize import FrameReader, head_prefix, int8_exclusions, quantize_int8
from tests.synthetic import make_synthetic_dataset

IMGSZ = 8


def tiny() -> onnx.ModelProto:
    """Giriş images [1,3,8,8] -> /model.0/conv/Conv -> Sigmoid -> /model.1/cv2/Conv -> Sigmoid."""
    rng = np.random.default_rng(0)
    w0 = rng.normal(0, 0.3, (4, 3, 3, 3)).astype(np.float32)
    w1 = rng.normal(0, 0.3, (6, 4, 1, 1)).astype(np.float32)
    nodes = [
        helper.make_node(
            "Conv", ["images", "w0"], ["a0"], name="/model.0/conv/Conv", pads=[1, 1, 1, 1]
        ),
        helper.make_node("Sigmoid", ["a0"], ["a1"], name="/model.0/act/Sigmoid"),
        helper.make_node("Conv", ["a1", "w1"], ["a2"], name="/model.1/cv2/Conv"),
        helper.make_node("Sigmoid", ["a2"], ["output0"], name="/model.1/Sigmoid"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 8, 8])],
        initializer=[
            helper.make_tensor("w0", TensorProto.FLOAT, [4, 3, 3, 3], w0.ravel().tolist()),
            helper.make_tensor("w1", TensorProto.FLOAT, [6, 4, 1, 1], w1.ravel().tolist()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    helper.set_model_props(model, {"channels": "3", "end2end": "True"})
    onnx.checker.check_model(model)
    return model


def _save(model: onnx.ModelProto, tmp_path: Path, name: str = "tiny.onnx") -> Path:
    path = tmp_path / name
    onnx.save(model, path)
    return path


def _sample(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((1, 3, IMGSZ, IMGSZ), dtype=np.float32)


class RandReader(CalibrationDataReader):
    """Seed'li n rastgele kalibrasyon girdisi; ilk örnek _sample(seed) ile aynıdır."""

    def __init__(self, n: int = 8, seed: int = 0) -> None:
        self._n = n
        self._seed = seed
        self._i = 0
        self._rng = np.random.default_rng(seed)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._i >= self._n:
            return None
        self._i += 1
        return {"images": self._rng.random((1, 3, IMGSZ, IMGSZ), dtype=np.float32)}

    def rewind(self) -> None:
        self._i = 0
        self._rng = np.random.default_rng(self._seed)


def _weight_is_dequantized(model: onnx.ModelProto, conv_name: str) -> bool:
    conv = next(n for n in model.graph.node if n.name == conv_name)
    producers = {o: n for n in model.graph.node for o in n.output}
    return conv.input[1] in producers and producers[conv.input[1]].op_type == "DequantizeLinear"


def test_head_prefix():
    assert head_prefix(tiny()) == "/model.1/"


def test_head_prefix_without_model_nodes_is_error():
    model = tiny()
    for node, name in zip(model.graph.node, ["conv", "sig", "conv2", "sig2"], strict=True):
        node.name = name
    with pytest.raises(ValueError):
        head_prefix(model)


def test_exclusions_follow_ultralytics_rule():
    assert int8_exclusions(tiny(), False) == ["/model.0/act/Sigmoid", "/model.1/Sigmoid"]
    assert int8_exclusions(tiny(), True) == [
        "/model.0/act/Sigmoid",
        "/model.1/Sigmoid",
        "/model.1/cv2/Conv",
    ]


def test_full_int8_quantizes_head_conv(tmp_path):
    src = _save(tiny(), tmp_path)
    dst = tmp_path / "tiny_int8.onnx"
    info = quantize_int8(src, dst, RandReader(), head_fp32=False)
    q = onnx.load(dst)
    assert _weight_is_dequantized(q, "/model.0/conv/Conv")
    assert _weight_is_dequantized(q, "/model.1/cv2/Conv")
    assert info["excluded_nodes"] == 2
    assert info["per_channel"] is True
    assert info["format"] == "QDQ"
    assert info["head_prefix"] == "/model.1/"
    assert info["calibration_images"] == 8


def test_head_fp32_keeps_head_conv_float(tmp_path):
    src = _save(tiny(), tmp_path)
    dst = tmp_path / "tiny_head.onnx"
    quantize_int8(src, dst, RandReader(), head_fp32=True)
    q = onnx.load(dst)
    assert _weight_is_dequantized(q, "/model.0/conv/Conv")
    assert not _weight_is_dequantized(q, "/model.1/cv2/Conv")


def test_quantized_model_runs_close_and_keeps_metadata(tmp_path):
    src = _save(tiny(), tmp_path)
    dst = tmp_path / "tiny_int8.onnx"
    quantize_int8(src, dst, RandReader(), head_fp32=False)
    x = _sample(0)
    providers = ["CPUExecutionProvider"]
    a = ort.InferenceSession(str(src), providers=providers).run(None, {"images": x})[0]
    b = ort.InferenceSession(str(dst), providers=providers).run(None, {"images": x})[0]
    assert np.abs(a - b).max() < 0.1
    props = {p.key: p.value for p in onnx.load(dst).metadata_props}
    assert props == {"channels": "3", "end2end": "True"}


def test_frame_reader_rewinds(tmp_path):
    root = make_synthetic_dataset(tmp_path / "d", with_ir=True)
    reader = FrameReader(root, "val", ["00101", "00103", "00104"], "ir", imgsz=64)
    for _ in range(3):
        batch = reader.get_next()
        assert set(batch) == {"images"}
        assert batch["images"].shape == (1, 3, 64, 64)
        assert batch["images"].dtype == np.float32
    assert reader.get_next() is None
    reader.rewind()
    for _ in range(3):
        assert reader.get_next()["images"].shape == (1, 3, 64, 64)
    assert reader.get_next() is None
