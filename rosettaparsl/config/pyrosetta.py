"""Configuration for pyRosetta initialization and minimization."""

from __future__ import annotations

import pyrosetta
from pydantic import Field

from rosettaparsl.utils import BaseModel


class PyRosettaConfig(BaseModel):
    """Configuration for PyRosetta initialization."""

    database_path: str = Field(
        ...,
        description='Path to the PyRosetta database',
    )
    weights_file: str = Field(..., description='Path to the weights file')
    iterations: int = Field(
        50,
        description='Number of iterations for minimization',
    )
    tolerance: float = Field(
        0.01,
        description='Convergence tolerance for minimization',
    )
    scorefxn: str = Field(
        'ref2015',
        description='Name of the score function to use',
    )
    minimization_method: str = Field(
        'lbfgs_armijo_nonmonotone',
        description='Minimization method to use',
    )

    # Get the score function based on the passed name.
    def get_scorefxn(self) -> pyrosetta.rosetta.core.scoring.ScoreFunction:
        """Get the score function based on the provided name."""
        if self.scorefxn == 'ref2015':
            return pyrosetta.get_fa_scorefxn()
        else:
            raise ValueError(f'Unknown score function: {self.scorefxn}')
