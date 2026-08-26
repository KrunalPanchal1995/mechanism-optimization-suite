from pathlib import Path

from mso_studio.core.nominal import NominalSimParams


def test_nominal_params_defaults_output_dir_and_uncertainty(tmp_path):
    targets = tmp_path / "t.target"
    targets.write_text("")
    params = NominalSimParams(
        label="nominal_htc",
        mechanism=str(tmp_path / "mech.yaml"),
        targets=str(targets),
        thermo_file=str(tmp_path / "thermo.dat"),
        trans_file=str(tmp_path / "trans.dat"),
        targets_count=5,
        base_opt_file=str(tmp_path / "base.opt"),
    )
    assert params.output_dir.endswith("SIM_RUNS/nominal_htc")
    # Nominal runs don't need real uncertainty data; defaults to targets path.
    assert params.uncertainty_data == str(targets)


def test_nominal_params_yaml_schema_matches_pipeline_config():
    params = NominalSimParams(
        label="nominal_htc",
        mechanism="/a/mech.yaml",
        targets="/a/t.target",
        thermo_file="/a/thermo.dat",
        trans_file="/a/trans.dat",
        targets_count=10,
        base_opt_file="/a/base.opt",
        output_dir="/a/out",
        uncertainty_data="/a/dummy.xml",
    )
    doc = params.to_sim_config_yaml()
    sim = doc["simulations"]["nominal_htc"]
    # Keys required by mso_compare.config.load_config's sim_configs loop.
    for key in ("mechanism", "targets", "targets_count", "thermo_file",
                "trans_file", "uncertainty_data", "output_dir"):
        assert key in sim
    assert sim["mechanism"] == "/a/mech.yaml"
    assert doc["base_opt_file"] == "/a/base.opt"
