from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


class DesignMatrixWriter:
    """
    Build and export diagonal matrices from a reaction uncertainty dictionary.

    Expected input
    --------------
    uncertainty_dict: dict[str, reaction_object]

    Each reaction_object must have:
        reaction_object.zeta.x

    where `zeta.x` is length 3:
        [A, n, Ea]

    For n reactions, every matrix written here is of size (3n x 3n).
    """

    def __init__(
        self,
        uncertainty_dict: Mapping[str, Any],
        perturbation_values: Sequence[float] = (1.0, 1.0, 200.0),
        output_dir: str | Path = ".",
    ) -> None:
        if len(perturbation_values) != 3:
            raise ValueError(
                "perturbation_values must contain exactly 3 values: (A, n, Ea)"
            )

        self.uncertainty_dict: Dict[str, Any] = dict(uncertainty_dict)
        self.perturbation_values = tuple(float(v) for v in perturbation_values)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reaction_names = list(self.uncertainty_dict.keys())

    def _extract_zeta_vector(self, rxn_name: str) -> np.ndarray:
        rxn_obj = self.uncertainty_dict[rxn_name]
        #print(rxn_name)
        try:
            zeta_x = rxn_obj.zeta.x
            #print(zeta_x)
        except AttributeError as exc:
            raise AttributeError(
                f"Reaction '{rxn_name}' does not have the expected attribute `zeta.x`."
            ) from exc

        zeta_array = np.asarray(zeta_x, dtype=float).reshape(-1)
        if zeta_array.size != 3:
            raise ValueError(
                f"Reaction '{rxn_name}' has zeta.x of size {zeta_array.size}, "
                "but size 3 was expected: [A, n, Ea]."
            )

        return zeta_array

    def _build_zero_matrix_from_vectors(
        self,
        vectors: Iterable[Sequence[float]],
    ) -> np.ndarray:
        """
        Build an (n x 3n) matrix where each row contains one 3-element vector
        in its own block position.
        """
        vectors = [np.asarray(v, dtype=float).reshape(3) for v in vectors]
        n = len(vectors)

        matrix = np.zeros((1, 3 * n), dtype=float)
        return matrix
    def _build_diagonal_matrix_from_vectors(
        self,
        vectors: Iterable[Sequence[float]],
    ) -> np.ndarray:
        """
        Build an (n x 3n) matrix where each row contains one 3-element vector
        in its own block position.
        """
        vectors = [np.asarray(v, dtype=float).reshape(3) for v in vectors]
        n = len(vectors)

        matrix = np.zeros((n, 3 * n), dtype=float)

        for i, vec in enumerate(vectors):
            matrix[i, 3 * i : 3 * i + 3] = vec

        return matrix

    def build_uncertainty_matrix(self) -> np.ndarray:
        zeta_vectors = [self._extract_zeta_vector(rxn_name) for rxn_name in self.reaction_names]
        
        return self._build_diagonal_matrix_from_vectors(zeta_vectors)

    def write_uncertainty_matrix(self, filename: str = "DesignMatrix_3P.csv") -> Path:
        matrix = self.build_uncertainty_matrix()
        filepath = self.output_dir / filename
        np.savetxt(filepath, matrix, delimiter=",", fmt="%.16g")
        return filepath

    def _build_single_parameter_converter(self, parameter_index: int) -> np.ndarray:
        perturb_vec = np.zeros(3, dtype=float)
        perturb_vec[parameter_index] = self.perturbation_values[parameter_index]

        vectors = [perturb_vec.copy() for _ in self.reaction_names]
        return self._build_diagonal_matrix_from_vectors(vectors)

    def write_converter_matrices(self) -> dict[str, Path]:
        file_map = {
            "convertor_A": (0, "convertor_A.csv"),
            "convertor_n": (1, "convertor_n.csv"),
            "convertor_Ea": (2, "convertor_Ea.csv"),
        }

        output_files: dict[str, Path] = {}

        for name, (param_index, filename) in file_map.items():
            matrix = self._build_single_parameter_converter(param_index)
            filepath = self.output_dir / filename
            np.savetxt(filepath, matrix, delimiter=",", fmt="%.16g")
            output_files[name] = filepath

        return output_files

    def build_design_matrix(self) -> np.ndarray:
        """
        Build a 3n x 3n diagonal matrix where each reaction contributes
        the vector (0, 0, 0) on the diagonal.
        """
        zero_vec = np.array([0.0, 0.0, 0.0])
        vectors = [zero_vec.copy() for _ in self.reaction_names]
        return self._build_zero_matrix_from_vectors(vectors)

    def write_design_matrix(self, filename: str = "DesignMatrix_x0_3P.csv") -> Path:
        matrix = self.build_design_matrix()
        filepath = self.output_dir / filename
        np.savetxt(filepath, matrix, delimiter=",", fmt="%.16g")
        return filepath

    def write_all(self) -> dict[str, Path]:
        outputs = {
            "uncertainty_matrix": self.write_uncertainty_matrix(),
            "DesignMatrix": self.write_design_matrix(),
        }
        outputs.update(self.write_converter_matrices())
        return outputs
