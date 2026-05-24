import json
from pathlib import Path

from group_photobooth.comfy_io import substitute_template


def test_substitute_replaces_string_placeholders():
    tpl = {"nodes": {"1": {"inputs": {"image": "$$IMAGE",
                                       "ignored": "no marker"}}}}
    out = substitute_template(tpl, {"$$IMAGE": "photo_abc.png"})
    assert out["nodes"]["1"]["inputs"]["image"] == "photo_abc.png"
    assert out["nodes"]["1"]["inputs"]["ignored"] == "no marker"


def test_substitute_replaces_inside_nested_lists():
    tpl = {"nodes": {"3": {"inputs": {"mask_or_coordinates": "$$BBOX_JSON"}}}}
    out = substitute_template(tpl, {"$$BBOX_JSON": "[[10,20,30,40]]"})
    assert out["nodes"]["3"]["inputs"]["mask_or_coordinates"] == "[[10,20,30,40]]"


def test_substitute_leaves_unknown_markers_alone():
    tpl = {"x": "$$UNKNOWN"}
    out = substitute_template(tpl, {"$$OTHER": "foo"})
    assert out["x"] == "$$UNKNOWN"
