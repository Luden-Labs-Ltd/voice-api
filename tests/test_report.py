"""
Tests for report module, especially JSON serialization.
"""

import json
import pytest
from readscore.report import convert_to_serializable


class TestConvertToSerializable:
    """Tests for numpy and non-standard type conversion."""

    def test_standard_types_unchanged(self):
        """Standard Python types should pass through unchanged."""
        assert convert_to_serializable(None) is None
        assert convert_to_serializable(True) is True
        assert convert_to_serializable(False) is False
        assert convert_to_serializable(42) == 42
        assert convert_to_serializable(3.14) == 3.14
        assert convert_to_serializable("hello") == "hello"

    def test_dict_conversion(self):
        """Dicts should be recursively converted."""
        data = {"a": 1, "b": {"c": 2}}
        result = convert_to_serializable(data)
        assert result == {"a": 1, "b": {"c": 2}}
        assert json.dumps(result)  # Should not raise

    def test_list_conversion(self):
        """Lists should be recursively converted."""
        data = [1, [2, 3], {"a": 4}]
        result = convert_to_serializable(data)
        assert result == [1, [2, 3], {"a": 4}]
        assert json.dumps(result)  # Should not raise

    def test_tuple_to_list(self):
        """Tuples should be converted to lists."""
        data = (1, 2, 3)
        result = convert_to_serializable(data)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_pathlib_path(self):
        """pathlib.Path should be converted to string."""
        from pathlib import Path
        data = {"path": Path("/some/path")}
        result = convert_to_serializable(data)
        assert result == {"path": "/some/path"}
        assert json.dumps(result)  # Should not raise

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_numpy_bool(self):
        """numpy.bool_ should be converted to Python bool."""
        import numpy as np
        data = {"flag": np.bool_(True), "flag2": np.bool_(False)}
        result = convert_to_serializable(data)
        assert result == {"flag": True, "flag2": False}
        assert isinstance(result["flag"], bool)
        assert json.dumps(result)  # Should not raise

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_numpy_integers(self):
        """numpy integer types should be converted to Python int."""
        import numpy as np
        data = {
            "int32": np.int32(42),
            "int64": np.int64(100),
            "uint8": np.uint8(255),
        }
        result = convert_to_serializable(data)
        assert result == {"int32": 42, "int64": 100, "uint8": 255}
        assert all(isinstance(v, int) for v in result.values())
        assert json.dumps(result)  # Should not raise

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_numpy_floats(self):
        """numpy float types should be converted to Python float."""
        import numpy as np
        data = {
            "float32": np.float32(3.14),
            "float64": np.float64(2.718),
        }
        result = convert_to_serializable(data)
        assert isinstance(result["float32"], float)
        assert isinstance(result["float64"], float)
        assert json.dumps(result)  # Should not raise

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_numpy_array(self):
        """numpy.ndarray should be converted to list."""
        import numpy as np
        data = {"array": np.array([1, 2, 3])}
        result = convert_to_serializable(data)
        assert result == {"array": [1, 2, 3]}
        assert isinstance(result["array"], list)
        assert json.dumps(result)  # Should not raise

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_nested_numpy_types(self):
        """Nested structures with numpy types should be fully converted."""
        import numpy as np
        data = {
            "scores": {
                "accuracy": np.float64(0.95),
                "flags": np.array([np.bool_(True), np.bool_(False)]),
            },
            "counts": [np.int64(1), np.int64(2)],
        }
        result = convert_to_serializable(data)

        # Verify structure
        assert isinstance(result["scores"]["accuracy"], float)
        assert isinstance(result["scores"]["flags"], list)
        assert all(isinstance(f, bool) for f in result["scores"]["flags"])
        assert all(isinstance(c, int) for c in result["counts"])

        # Must serialize without error
        json_str = json.dumps(result)
        assert json_str  # Non-empty valid JSON

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy not installed"
    )
    def test_report_like_structure(self):
        """Simulate a report structure with numpy types."""
        import numpy as np
        report = {
            "input": {
                "audio": "test.wav",
                "text_len_words": np.int64(10),
                "duration_sec": np.float32(5.5),
            },
            "accuracy": {
                "wer": np.float64(0.1),
                "counts": {
                    "ins": np.int32(0),
                    "del": np.int32(1),
                    "sub": np.int32(0),
                },
            },
            "fluency_speed": {
                "wpm": np.float64(150.0),
                "within_range": np.bool_(True),
            },
            "prosody": {
                "score_0_100": np.float32(85.0),
                "flags": [],
            },
            "pronunciation_quality": {
                "score_0_100": np.float64(90.0),
            },
        }

        result = convert_to_serializable(report)

        # Verify all values are serializable
        json_str = json.dumps(result, indent=2)
        assert json_str

        # Verify structure preserved
        parsed = json.loads(json_str)
        assert parsed["input"]["text_len_words"] == 10
        assert parsed["fluency_speed"]["within_range"] is True
        assert parsed["accuracy"]["wer"] == pytest.approx(0.1)

    def test_unknown_object_to_string(self):
        """Unknown objects should be converted to their string representation."""

        class CustomObject:
            def __str__(self):
                return "custom_value"

        data = {"obj": CustomObject()}
        result = convert_to_serializable(data)
        assert result == {"obj": "custom_value"}
        assert json.dumps(result)  # Should not raise
