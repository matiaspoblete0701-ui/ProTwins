# ProTwins: Protein Structural Hierarchy and Functional Annotation Pipeline

**ProTwins** is a command-line tool designed for high-throughput structural comparison, hierarchical clustering, and functional annotation of protein structures (`.pdb` or `.cif`). It integrates the precision of **USalign** for pairwise structural alignments with the speed of **Foldseek** for cloud-based database homology searches.

## Key Features

- **Automated Pairwise Alignment:** Calculates comprehensive structural similarity matrices (TM-scores) using local `USalign` execution.
- **Hierarchical Clustering:** Groups proteins based on structural distance ($1 - \text{TM-score}$) using the average linkage method.
- **Advanced Visualizations:** Automatically generates publication-ready Heatmaps, Clustermaps, and Dendrograms partitioned by user-defined cutting thresholds.
- **Representative Medoid Selection:** Identifies the exact medoid for each structural cluster to capture representative topologies.
- **PyMOL Integration:** Generates customized `.pml` visualization scripts with distance-based color gradients for multi-structure alignment.
- **Functional Annotation:** Connects to the Foldseek API to screen query structures or cluster medoids against global structural databases (PDB, AlphaFold DB, Swissprot, etc.), generating consolidated Excel/CSV reports.

## Requirements & Installation

### 1. Clone the Repository
git clone (https://github.com/matiaspoblete0701-ui/ProTwins)
cd ProTwins

