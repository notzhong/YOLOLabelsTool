"""共享测试 fixtures"""

from pathlib import Path

import pytest
from src.core.annotation import Annotation, AnnotationManager
from src.core.class_manager import ClassManager


@pytest.fixture
def sample_annotation():
    return Annotation(x=10, y=20, width=100, height=200, class_id=0)


@pytest.fixture
def sample_annotations():
    return [
        Annotation(x=10, y=20, width=100, height=200, class_id=0),
        Annotation(x=50, y=60, width=80, height=120, class_id=1),
        Annotation(x=100, y=150, width=60, height=90, class_id=0),
    ]


@pytest.fixture
def annotation_manager(tmp_path):
    mgr = AnnotationManager()
    mgr._annotation_dir = str(tmp_path / "annotations")
    Path(mgr._annotation_dir).mkdir(parents=True, exist_ok=True)
    return mgr


@pytest.fixture
def class_manager():
    return ClassManager()


@pytest.fixture
def class_manager_with_classes(class_manager):
    cm = class_manager
    cm.add_class("person", (255, 0, 0))
    cm.add_class("car", (0, 255, 0))
    cm.add_class("dog", (0, 0, 255))
    return cm
