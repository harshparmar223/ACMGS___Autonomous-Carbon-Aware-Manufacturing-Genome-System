"""
Batch Genome Package
Unified 25-D Feature Fusion Engine combining Process, Material, 16-D Energy DNA, and Grid Carbon signals.
"""

from src.batch_genome.encoder import BatchGenomeEncoder, build_and_save_genome_dataset

__all__ = ["BatchGenomeEncoder", "build_and_save_genome_dataset"]
