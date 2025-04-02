"""Run Rosetta to predict complex stability."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import os.path as osp

import pyrosetta
from pyrosetta import rosetta
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from rosettaparsl.parsl import ComputeConfigs
from rosettaparsl.utils import BaseModel
from rosettaparsl.utils import batch_data

# need abstraction for a pdb?


class PyRosettaConfig(BaseModel):
    """Configuration for PyRosetta initialization."""

    database_path: str = Field(
        ..., description='Path to the PyRosetta database'
    )
    weights_file: str = Field(..., description='Path to the weights file')
    iterations: int = Field(
        50, description='Number of iterations for minimization'
    )
    tolerance: float = Field(
        0.01, description='Convergence tolerance for minimization'
    )
    scorefxn: str = Field(
        'ref2015', description='Name of the score function to use'
    )
    minimization_method: str = Field(
        'lbfgs_armijo_nonmonotone',
        description='Minimization method to use',
    )

    # initialize pyrosetta
    @model_validator(mode='after')
    def _init_pyrosetta(self) -> Self:
        """Initialize PyRosetta with the provided configuration."""
        pyrosetta.init(
            f'-database {self.database_path} -ex1 -ex2aro -use_input_sc -ignore_unrecognized_res -score:weights {self.weights_file}'
        )
        return self

    # Get the score function based on the passed name.
    def get_scorefxn(self) -> pyrosetta.rosetta.core.scoring.ScoreFunction:
        """Get the score function based on the provided name."""
        if self.scorefxn == 'ref2015':
            return pyrosetta.get_fa_scorefxn()
        else:
            raise ValueError(f'Unknown score function: {self.scorefxn}')


class RosettaParslWorkflowConfig(BaseModel):
    """Configuration for the RosettaParsl workflow."""

    input_dir: str = Field(..., description='Directory containing PDB files')
    output_file: str = Field(..., description='CSV file to save results')
    chunk_size: int = Field(
        1, description='Number of PDB files to process in each worker call'
    )
    pyrosetta_config: PyRosettaConfig = Field(
        ...,
        description='Configuration for PyRosetta initialization'
        ' and minimization',
    )
    compute_config: ComputeConfigs = Field(
        ...,
        description='Configuration for the Parsl compute resources',
    )

    @model_validator(mode='after')
    def _validate_paths(self) -> Self:
        # Check if the paths exist.
        if not Path(self.input_dir).exists():
            raise ValueError(
                f'Input directory {self.input_dir} does not exist.'
            )

        # Create the parent directory of the output file if it does not exist.
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve the paths.
        self.input_dir = Path(self.input_dir).resolve()
        self.output_file = output_path.resolve()

        return self


def rosettaparsl_worker(
    pdb_files: list[str], output_dir: Path, pyrosetta_config: PyRosettaConfig
) -> None:
    """Process PDB files: pack side chains, minimize, and compute stability."""

    # Initialize the scoring function
    score_fxn = pyrosetta_config.get_scorefxn()

    # Load the structure
    for pdb_file in pdb_files:
        try:
            pose = pyrosetta.pose_from_pdb(pdb_file)
        except Exception as e:
            print(f'Error loading PDB file {pdb_file}: {e}')
            continue

        # Create task and pack side chains
        task_factory = rosetta.core.pack.task.TaskFactory()
        task = task_factory.create_packer_task(pose)
        task.restrict_to_repacking()

        packer = rosetta.protocols.minimization_packing.PackRotamersMover()
        packer.score_function(score_fxn)
        packer.task_factory(task_factory)

        # Pack the side chains
        try:
            packer.apply(pose)
        except Exception as e:
            print(f'Error during side chain packing for {pdb_file}: {e}')
            continue

        # Backbone & side-chain minimization using LBFGS
        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        min_mover.score_function(score_fxn)
        min_mover.min_type(pyrosetta_config.minimization_method)
        min_mover.tolerance(pyrosetta_config.tolerance)

        prev_score = score_fxn(pose)
        for _ in range(pyrosetta_config.iterations):
            min_mover.apply(pose)
            new_score = score_fxn(pose)
            if abs(new_score - prev_score) < pyrosetta_config.tolerance:
                break
            prev_score = new_score

        # Compute the stability score (ΔΔG)
        E_folded = score_fxn(pose)
        stability_score = E_folded

        if stability_score is None:
            return None
        pdb_id = osp.basename(pdb_file).split('_')[0]

        # Write the stability score to a file
        with open(output_dir / f'{pdb_id}_stability.txt', 'w') as f:
            f.write(f'{stability_score}\n')
