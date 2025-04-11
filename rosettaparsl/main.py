"""Run Rosetta to predict complex stability."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from rosettaparsl.config import PyRosettaConfig
from rosettaparsl.parsl import ComputeConfigs
from rosettaparsl.utils import BaseModel
from rosettaparsl.utils import batch_data


class RosettaParslWorkflowConfig(BaseModel):
    """Configuration for the RosettaParsl workflow."""

    input_dir: Path = Field(..., description='Directory containing PDB files')
    output_dir: Path = Field(..., description='Directory to save CSV results.')
    chunk_size: int = Field(
        1,
        description='Number of PDB files to process in each worker call',
    )
    pyrosetta_config: PyRosettaConfig = Field(
        ...,
        description='Config for PyRosetta initialization and minimization',
    )
    compute_config: ComputeConfigs = Field(
        ...,
        description='Configuration for the Parsl compute resources',
    )

    @model_validator(mode='after')
    def _validate_paths(self) -> Self:
        # Check if the paths exist.
        if not self.input_dir.exists():
            raise ValueError(
                f'Input directory {self.input_dir} does not exist.',
            )

        # Create the output directory if it does not exist.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the paths.
        self.input_dir = self.input_dir.resolve()
        self.output_dir = self.output_dir.resolve()

        return self


def rosettaparsl_worker(
    pdb_files: list[Path],
    output_dir: Path,
    pyrosetta_config: PyRosettaConfig,
) -> None:
    """Process PDB files: pack side chains, minimize, and compute stability."""
    import pyrosetta
    from pyrosetta import rosetta

    # Reinitialize PyRosetta in the worker process
    try:
        pyrosetta.init(
            f'-database {pyrosetta_config.database_path} -ex1 -ex2aro -use_input_sc '  # noqa: E501
            f'-ignore_unrecognized_res -score:weights {pyrosetta_config.weights_file}',  # noqa: E501
        )
    except Exception as init_err:
        print(f'Error initializing PyRosetta in worker: {init_err}')
        return

    # Initialize the scoring function
    score_fxn = pyrosetta_config.get_scorefxn()

    # Process each PDB file
    for pdb_file in pdb_files:
        try:
            pose = pyrosetta.pose_from_pdb(str(pdb_file))
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

        # Backbone & side-chain minimization using the configured method
        min_mover = rosetta.protocols.minimization_packing.MinMover()
        min_mover.score_function(score_fxn)
        min_mover.min_type(pyrosetta_config.minimization_method)
        min_mover.tolerance(pyrosetta_config.tolerance)

        # Iteratively minimize the energy (maximize stability score)
        prev_score = score_fxn(pose)
        for _ in range(pyrosetta_config.iterations):
            min_mover.apply(pose)
            new_score = score_fxn(pose)
            if abs(new_score - prev_score) < pyrosetta_config.tolerance:
                break
            prev_score = new_score

        # Compute the stability score (ΔΔG);
        # here the folded energy is used as the stability score.
        folded_energy = score_fxn(pose)
        stability_score = folded_energy

        # In case of error, skip this file rather than exiting worker
        if stability_score is None:
            continue

        pdb_id = pdb_file.stem.split('_')[1]

        # Write the stability score to an individual file
        output_file = output_dir / f'{pdb_id}_stability.txt'
        try:
            with open(output_file, 'w') as f:
                f.write(f'{stability_score}\n')
        except Exception as write_err:
            print(f'Error writing results for {pdb_file}: {write_err}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run RosettaParsl workflow to predict protein stability.',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file for RosettaParsl.',
    )
    args = parser.parse_args()

    # Load the configuration from the YAML file
    config = RosettaParslWorkflowConfig.from_yaml(args.config)

    # Dump the configuration for record keeping
    config.dump_yaml(config.output_dir / 'rosettaparsl_config.yaml')

    # Set the Parsl compute settings.
    parsl_config = config.compute_config.get_parsl_config(
        config.output_dir / 'parsl',
    )

    # Collect PDB files from the input directory.
    if config.input_dir.is_dir():
        pdb_files = list(config.input_dir.glob('*.pdb'))
        if not pdb_files:
            raise ValueError(
                f'No PDB files found in input directory {config.input_dir}',
            )
    else:
        raise ValueError(
            f'Input path {config.input_dir} is not a directory. '
            'Even if you have a single PDB file, pass the parent directory.',
        )

    # Batch the PDB files based on the chunk size.
    pdb_batches = batch_data(pdb_files, config.chunk_size)

    # Define the worker function with fixed arguments.
    worker_fn = functools.partial(
        rosettaparsl_worker,
        output_dir=config.output_dir / 'rosetta_results',
        pyrosetta_config=config.pyrosetta_config,
    )

    # Distribute the Rosetta calculations over workers.
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, pdb_batches))
