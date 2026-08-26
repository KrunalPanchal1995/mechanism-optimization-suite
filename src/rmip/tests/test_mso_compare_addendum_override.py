"""
rmip/tests/test_mso_compare_addendum_override.py
====================================================
Regression coverage for Phase 5b (2026-08 backlog program, bug.md):
SimRunConfig.addendum, a new optional per-run .add path.

Before this, Locations.addendum was only ever inherited from whatever the
run's base_opt_file template already specified -- there was no way for a
run's own config to point at a DIFFERENT addendum than its template's,
which blocked wiring src/mso_io/target_add_io.py's preview/edit primitives
into nominal_sim_gui.py (KRUNAL_JULY_23.md point 4's "same for nominal
simulations" follow-up): previewing/authoring targets for a run requires
knowing which .add file that specific run will actually use, and until now
that could only ever be the template's, never overridden per run.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mso_compare.config import SimRunConfig
from mso_compare.utils import build_run_config


def _make_sim_run_cfg(**overrides):
    defaults = dict(
        label="test_run", mechanism="mech.yaml", targets="t.target",
        targets_count=1, thermo_file="t.therm", trans_file="t.trans",
        uncertainty_data="", output_dir="/tmp/out",
    )
    defaults.update(overrides)
    return SimRunConfig(**defaults)


class TestAddendumOverride:
    def test_unset_addendum_leaves_base_opt_untouched(self):
        """The common case: no addendum set on the run config -- the base
        .opt template's own Locations.addendum must survive unchanged."""
        base_opt = {"Locations": {"addendum": "template.add", "targets": "old.target"}}
        cfg = _make_sim_run_cfg()  # addendum defaults to None
        result = build_run_config(base_opt, cfg)
        assert result["Locations"]["addendum"] == "template.add"

    def test_set_addendum_overrides_base_opt(self):
        base_opt = {"Locations": {"addendum": "template.add"}}
        cfg = _make_sim_run_cfg(addendum="/path/to/run_specific.add")
        result = build_run_config(base_opt, cfg)
        assert result["Locations"]["addendum"] == "/path/to/run_specific.add"

    def test_addendum_override_coexists_with_other_locations_fields(self):
        """The addendum override must not clobber the other Locations
        overrides build_run_config already applies (mechanism/targets/etc)."""
        base_opt = {"Locations": {"addendum": "template.add", "mechanism": "old.yaml"}}
        cfg = _make_sim_run_cfg(mechanism="new.yaml", addendum="new.add")
        result = build_run_config(base_opt, cfg)
        assert result["Locations"]["addendum"] == "new.add"
        assert result["Locations"]["mechanism"] == "new.yaml"

    def test_base_opt_with_no_addendum_key_and_unset_override(self):
        """Neither the template nor the run config specify an addendum --
        must not KeyError or fabricate one."""
        base_opt = {"Locations": {}}
        cfg = _make_sim_run_cfg()
        result = build_run_config(base_opt, cfg)
        assert "addendum" not in result["Locations"]

    def test_extra_overrides_still_take_precedence_over_addendum_field(self):
        """extra_overrides is documented to apply LAST -- confirm it can
        still override Locations.addendum even when the addendum field is
        also set, matching the existing precedence contract for every other
        field."""
        base_opt = {"Locations": {"addendum": "template.add"}}
        cfg = _make_sim_run_cfg(
            addendum="field.add",
            extra_overrides={"Locations": {"addendum": "extra_override.add"}},
        )
        result = build_run_config(base_opt, cfg)
        assert result["Locations"]["addendum"] == "extra_override.add"

    def test_original_base_opt_dict_not_mutated(self):
        base_opt = {"Locations": {"addendum": "template.add"}}
        cfg = _make_sim_run_cfg(addendum="new.add")
        build_run_config(base_opt, cfg)
        assert base_opt["Locations"]["addendum"] == "template.add"


class TestPreviewTargetsForRun:
    """sim_runner.preview_targets_for_run -- real end-to-end against actual
    .target/.add files written via mso_io.target_add_io's own primitives."""

    def _write_base_opt(self, tmp_path, targets_path, addendum_path):
        import yaml
        base_opt_path = tmp_path / "base.opt"
        base_opt_path.write_text(yaml.safe_dump({
            "Locations": {"targets": str(targets_path), "addendum": str(addendum_path)},
            "Counts": {"targets_count": 5},
        }))
        return str(base_opt_path)

    def test_previews_using_the_template_addendum_when_unset(self, tmp_path):
        from mso_io.target_add_io import append_case
        from mso_io.target_add_schema import NewCaseFields
        from mso_compare.sim_runner import preview_targets_for_run

        target_path = tmp_path / "template.target"
        add_path = tmp_path / "template.add"
        append_case(str(target_path), str(add_path),
                    NewCaseFields(dataset_id="template_case", solver="cantera"))

        base_opt_file = self._write_base_opt(tmp_path, target_path, add_path)
        cfg = _make_sim_run_cfg(targets=str(target_path))  # addendum unset

        rows = preview_targets_for_run(cfg, base_opt_file)
        assert len(rows) == 1
        assert rows[0].dataset_id == "template_case"
        assert rows[0].solver == "cantera"

    def test_previews_using_the_run_specific_addendum_when_set(self, tmp_path):
        """The core Phase 5b gain: a run's own addendum override actually
        changes which .add file the preview reads, not just the template's."""
        from mso_io.target_add_io import append_case
        from mso_io.target_add_schema import NewCaseFields
        from mso_compare.sim_runner import preview_targets_for_run

        target_path = tmp_path / "shared.target"
        template_add = tmp_path / "template.add"
        run_specific_add = tmp_path / "run_specific.add"

        # Same target file, but two DIFFERENT .add files disagreeing on solver.
        append_case(str(target_path), str(template_add),
                    NewCaseFields(dataset_id="case1", solver="cantera"))
        from mso_io.target_add_io import load_add, save_add
        save_add(str(run_specific_add), {"case1": {"solver": "CHEMKIN_PRO"}})

        base_opt_file = self._write_base_opt(tmp_path, target_path, template_add)
        cfg = _make_sim_run_cfg(targets=str(target_path), addendum=str(run_specific_add))

        rows = preview_targets_for_run(cfg, base_opt_file)
        assert len(rows) == 1
        assert rows[0].solver == "CHEMKIN_PRO"   # from run_specific_add, not template
