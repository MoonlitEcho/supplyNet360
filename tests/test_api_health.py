import sys
import types

from fastapi.testclient import TestClient


class _FakeModel:
    def predict(self, _input):
        return [1.0]


def _install_stubs() -> None:
    fake_joblib = types.ModuleType("joblib")
    fake_joblib.load = lambda path: (
        ["f1", "f2"] if str(path).endswith("_features.pkl") else _FakeModel()
    )

    class _FakeDataFrame:
        empty = True

        def __getitem__(self, _key):
            return self

        def mean(self):
            return {}

        def to_dict(self):
            return {}

        def sample(self, _n):
            return self

    fake_pandas = types.ModuleType("pandas")
    fake_pandas.read_csv = lambda *args, **kwargs: _FakeDataFrame()
    fake_pandas.DataFrame = lambda *args, **kwargs: _FakeDataFrame()
    fake_pandas.to_datetime = lambda value: value

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.mean = lambda values: sum(values) / len(values) if values else 0
    fake_numpy.abs = abs
    fake_numpy.sum = sum

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *args, **kwargs: None

    fake_shap = types.ModuleType("shap")
    fake_shap.TreeExplainer = lambda *_args, **_kwargs: None

    sys.modules.setdefault("joblib", fake_joblib)
    sys.modules.setdefault("pandas", fake_pandas)
    sys.modules.setdefault("numpy", fake_numpy)
    sys.modules.setdefault("uvicorn", fake_uvicorn)
    sys.modules.setdefault("shap", fake_shap)


_install_stubs()
from Deploy.api import app  # noqa: E402


def test_health_endpoint_returns_service_status() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "loaded_models" in payload
