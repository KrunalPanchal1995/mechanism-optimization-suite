from mso_studio.core.digitizer import (
    CalibPoint, DataPoint, export_csv, export_rkd_xml, export_yaml, pixel_to_data,
)


def test_pixel_to_data_linear_mapping():
    x_cal = (CalibPoint(px=10, py=0, value=0.0), CalibPoint(px=110, py=0, value=100.0))
    y_cal = (CalibPoint(px=0, py=200, value=0.0), CalibPoint(px=0, py=100, value=50.0))
    x, y = pixel_to_data(60, 150, x_cal, y_cal)
    assert x == 50.0
    assert y == 25.0


def test_pixel_to_data_degenerate_calibration_returns_nan():
    x_cal = (CalibPoint(px=10, py=0, value=0.0), CalibPoint(px=10, py=0, value=100.0))
    y_cal = (CalibPoint(px=0, py=0, value=0.0), CalibPoint(px=0, py=10, value=50.0))
    x, y = pixel_to_data(10, 5, x_cal, y_cal)
    assert x != x  # NaN


def test_export_csv_without_error_bars(tmp_path):
    points = [DataPoint(px=0, py=0, x=1.0, y=2.0), DataPoint(px=1, py=1, x=3.0, y=4.0)]
    out = tmp_path / "data.csv"
    export_csv(points, str(out), x_label="T_K", y_label="tau_ms")
    text = out.read_text()
    assert text.splitlines()[0] == "T_K,tau_ms"
    assert "1.0,2.0" in text


def test_export_csv_with_error_bars(tmp_path):
    points = [DataPoint(px=0, py=0, x=1.0, y=2.0, dy_plus=0.1, dy_minus=0.2)]
    out = tmp_path / "data.csv"
    export_csv(points, str(out))
    header = out.read_text().splitlines()[0]
    assert header == "x,y,y_plus,y_minus"


def test_export_rkd_xml(tmp_path):
    points = [DataPoint(px=0, py=0, x=1200.0, y=0.55)]
    out = tmp_path / "data.xml"
    export_rkd_xml(points, str(out))
    text = out.read_text()
    assert "<dataPoint>" in text
    assert "1200" in text


def test_export_yaml(tmp_path):
    points = [DataPoint(px=0, py=0, x=1.5, y=2.5)]
    out = tmp_path / "data.yaml"
    export_yaml(points, str(out), x_label="phi", y_label="idt")
    text = out.read_text()
    assert "x_label: phi" in text
    assert "phi: 1.5" in text
