#!/bin/bash

# Create output directories if they don't exist
mkdir -p stats logs

# Create header for stats file
echo -e "chr\tfile_size\ttime_seconds\tpeak_memory_gb" > stats/hgdp_haplo_stats.tsv

# Default to chromosome Y if no argument is provided
chr=${1:-Y}

# Submit job for the specified chromosome
qsub -N "hgdp_haplo_chr${chr}" run_hgdp_haplo_analysis.sh $chr

echo "HGDP chromosome ${chr} job submitted. Monitor with qstat" 