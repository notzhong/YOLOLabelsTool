"""src/core/model_manager.py 单元测试（使用假模型，无需真实 YOLO 权重）"""

import numpy as np
import pytest

import src.core.model_manager as mm_module
from src.core.model_manager import ModelManager


class FakeBox:
    """模拟 ultralytics 的 box 对象"""
    def __init__(self, x1, y1, x2, y2, conf, cls_id):
        class _T:
            def __init__(self, value):
                self.value = value
            def tolist(self):
                return self.value
            def item(self):
                return self.value
        self.xyxy = [_T([x1, y1, x2, y2])]
        self.conf = [_T(conf)]
        self.cls = [_T(cls_id)]


class FakeResult:
    def __init__(self, boxes, orig_shape=(480, 640)):
        self.boxes = boxes
        self.orig_shape = orig_shape


class FakeYOLOModel:
    """模拟 YOLO 模型：可调用、带 names 属性"""
    def __init__(self, results=None, names=None):
        self.names = names if names is not None else {0: "person", 1: "car"}
        self.results = results or []
        self.calls = []

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return self.results


class FakeYOLOCtor:
    """替代 ultralytics.YOLO 构造器"""
    def __init__(self, path):
        self.path = path
        self.names = {0: "person", 1: "car"}


@pytest.fixture
def mm():
    return ModelManager()


class TestInitAndThresholds:
    def test_initial_state(self, mm):
        assert mm.model is None
        assert mm.model_loaded is False
        assert mm.confidence_threshold == 0.25
        assert mm.iou_threshold == 0.45
        assert mm.class_names == {}

    def test_is_available_matches_module_flag(self, mm):
        assert mm.is_available() == mm_module.YOLO_AVAILABLE

    def test_confidence_threshold_clamped(self, mm):
        mm.set_confidence_threshold(1.5)
        assert mm.confidence_threshold == 1.0
        mm.set_confidence_threshold(-0.1)
        assert mm.confidence_threshold == 0.0

    def test_iou_threshold_clamped(self, mm):
        mm.set_iou_threshold(2.0)
        assert mm.iou_threshold == 1.0
        mm.set_iou_threshold(-1)
        assert mm.iou_threshold == 0.0


class TestLoadModel:
    def test_load_success(self, mm, monkeypatch, tmp_path):
        monkeypatch.setattr(mm_module, "YOLO", FakeYOLOCtor)
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake")

        assert mm.load_model(str(model_file)) is True
        assert mm.model_loaded is True
        assert mm.model_path == str(model_file.resolve())
        assert mm.class_names == {0: "person", 1: "car"}

    def test_load_invalid_file_returns_false(self, mm, monkeypatch):
        def boom(path):
            raise RuntimeError("bad model")

        monkeypatch.setattr(mm_module, "YOLO", boom)
        assert mm.load_model("/nonexistent.pt") is False
        assert mm.model_loaded is False


class TestProcessBoxes:
    def test_normal_box(self, mm):
        mm.class_names = {0: "person"}
        dets = mm._process_boxes([FakeBox(10, 20, 110, 220, 0.9, 0)], 640, 480)
        assert len(dets) == 1
        d = dets[0]
        assert (d["x"], d["y"], d["width"], d["height"]) == (10, 20, 100, 200)
        assert d["confidence"] == 0.9
        assert d["class_id"] == 0
        assert d["class_name"] == "person"

    def test_boxes_clipped_to_image(self, mm):
        dets = mm._process_boxes([FakeBox(-10, -10, 2000, 2000, 0.9, 0)], 100, 80)
        d = dets[0]
        assert (d["x"], d["y"]) == (0, 0)
        assert (d["width"], d["height"]) == (100, 80)

    def test_tiny_boxes_filtered(self, mm):
        dets = mm._process_boxes(
            [FakeBox(0, 0, 3, 100, 0.9, 0), FakeBox(0, 0, 100, 4, 0.9, 0)], 640, 480
        )
        assert dets == []

    def test_unknown_class_gets_fallback_name(self, mm):
        dets = mm._process_boxes([FakeBox(0, 0, 50, 50, 0.9, 7)], 640, 480)
        assert dets[0]["class_name"] == "class_7"


class TestPredict:
    def test_predict_without_model_returns_empty(self, mm, tmp_path):
        assert mm.predict("x.png") == []
        assert mm.predict_image(np.zeros((10, 10, 3), dtype=np.uint8)) == []

    def test_predict_with_fake_model(self, mm, tmp_path, real_image_factory):
        img = real_image_factory("p.png")
        mm.model = FakeYOLOModel(results=[FakeResult([FakeBox(5, 5, 45, 35, 0.8, 1)])])
        mm.model_loaded = True
        mm.class_names = mm.model.names

        dets = mm.predict(img)
        assert len(dets) == 1
        assert dets[0]["class_id"] == 1
        # 阈值参数被传递
        _, kwargs = mm.model.calls[0]
        assert kwargs["conf"] == 0.25 and kwargs["iou"] == 0.45

    def test_predict_skips_none_boxes(self, mm, real_image_factory):
        img = real_image_factory("p2.png")
        mm.model = FakeYOLOModel(results=[FakeResult(None)])
        mm.model_loaded = True
        assert mm.predict(img) == []

    def test_predict_image_uses_array_shape(self, mm):
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        mm.model = FakeYOLOModel(results=[FakeResult([FakeBox(0, 0, 100, 100, 0.9, 0)])])
        mm.model_loaded = True

        dets = mm.predict_image(arr)
        assert dets[0]["width"] == 100

    def test_predict_exception_returns_empty(self, mm, real_image_factory):
        img = real_image_factory("p3.png")

        class BoomModel:
            def __call__(self, *a, **k):
                raise RuntimeError("inference failed")

        mm.model = BoomModel()
        mm.model_loaded = True
        assert mm.predict(img) == []


class TestModelInfoAndLifecycle:
    def test_model_info_not_loaded(self, mm):
        info = mm.get_model_info()
        assert info == {"loaded": False, "path": None, "class_count": 0}

    def test_model_info_loaded(self, mm):
        mm.model_loaded = True
        mm.model_path = "/m.pt"
        mm.class_names = {0: "a", 1: "b"}
        info = mm.get_model_info()
        assert info["loaded"] is True
        assert info["class_count"] == 2
        assert info["confidence_threshold"] == 0.25

    def test_is_model_loaded(self, mm):
        assert mm.is_model_loaded() is False
        mm.model_loaded = True
        assert mm.is_model_loaded() is True

    def test_unload_resets_state(self, mm):
        mm.model = FakeYOLOModel()
        mm.model_loaded = True
        mm.model_path = "/m.pt"
        mm.class_names = {0: "a"}

        mm.unload_model()
        assert mm.model is None
        assert mm.model_path is None
        assert mm.model_loaded is False
        assert mm.class_names == {}


class TestConvertToAnnotations:
    def test_conversion_fields(self, mm):
        detections = [
            {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0, "class_id": 2, "confidence": 0.5},
        ]
        anns = mm.convert_to_annotations(detections)
        assert len(anns) == 1
        a = anns[0]
        assert (a.x, a.y, a.width, a.height, a.class_id) == (1.0, 2.0, 3.0, 4.0, 2)

    def test_empty_detections(self, mm):
        assert mm.convert_to_annotations([]) == []
