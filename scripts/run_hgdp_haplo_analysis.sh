#!/bin/bash
#$ -cwd
#$ -l m_mem_free=700G
#$ -N hgdp_haplo_analysis
#$ -j y
#$ -o ./logs

# Create output directories
mkdir -p stats logs

# Function to download file if it doesn't exist
download_file() {
    local url=$1
    local output=$2
    if [ ! -f "$output" ]; then
        echo "Downloading $url to $output"
        wget -O "$output" "$url"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download $url"
            exit 1
        fi
    else
        echo "File $output already exists, skipping download"
    fi
}

# Function to get file size in GB
get_file_size() {
    local file=$1
    ls -lh "$file" | awk '{print $5}'
}

# Function to process a single chromosome
process_chromosome() {
    local chr=$1
    local vcf_file=$2
    local output_file="hgdp_output_${chr}.json"
    
    # Start time
    start_time=$(date +%s)
    
    echo "Starting haplo analysis for chromosome ${chr}"
    echo "Input file: ${vcf_file}"
    echo "Output file: ${output_file}"
    
    # Run haplo with force_overwrite flag
    singularity exec --bind /grid/mccandlish/home/msun/test_haplo:/input \
        --bind $HOME/vep_data:/opt/vep/.vep ensembl-vep.sif \
        haplo -i "/input/${vcf_file}" \
        -o "/input/${output_file}" --json --cache --assembly GRCh38 --force_overwrite 2>&1 | tee "logs/haplo_chr${chr}.log"
    
    haplo_exit_code=${PIPESTATUS[0]}
    if [ $haplo_exit_code -ne 0 ]; then
        echo "Error: haplo analysis failed with exit code $haplo_exit_code"
        exit 1
    fi
    
    # Calculate elapsed time
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    
    # Get file size
    file_size=$(get_file_size "$vcf_file")
    
    # Record stats with elapsed time for now
    echo -e "${chr}\t${file_size}\t${elapsed_time}\tNA\tNA\tNA" >> stats/hgdp_haplo_stats.tsv
    
    # Clean up
    echo "Cleaning up temporary files"
    rm -f "$vcf_file" "${vcf_file}.tbi"
}

# Create header for stats file
echo -e "chr\tfile_size\ttime_seconds\tvirtual_memory_gb\tphysical_memory_mb\tgrid_memory_units" > stats/hgdp_haplo_stats.tsv

# Base URL for HGDP downloads
BASE_URL="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516"

# Process a single chromosome
chr=$1
if [ -z "$chr" ]; then
    echo "Error: No chromosome specified"
    echo "Usage: $0 <chromosome>"
    echo "Example: $0 Y"
    exit 1
fi

echo "Processing HGDP chromosome ${chr}..."

# Download VCF and TBI files
vcf_file="hgdp_wgs.20190516.full.chr${chr}.vcf.gz"
tbi_file="${vcf_file}.tbi"

echo "Downloading files for chromosome ${chr}"
download_file "${BASE_URL}/${vcf_file}" "$vcf_file"
download_file "${BASE_URL}/${tbi_file}" "$tbi_file"

# Verify files exist and have size
if [ ! -s "$vcf_file" ]; then
    echo "Error: VCF file is empty or does not exist"
    exit 1
fi

if [ ! -s "$tbi_file" ]; then
    echo "Error: TBI file is empty or does not exist"
    exit 1
fi

# Run haplo analysis
process_chromosome "$chr" "$vcf_file"

echo "Completed HGDP chromosome ${chr}"

# Store the current job ID
HAPLO_JOB_ID=$JOB_ID
echo "====================================================="
echo "Haplo analysis job ID: $HAPLO_JOB_ID"
echo "====================================================="

# Create a script to collect resource usage
cat > collect_stats.sh << EOF
#!/bin/bash
echo "====================================================="
echo "Stats collection job started for haplo job $HAPLO_JOB_ID"
echo "====================================================="

# Wait for the haplo job to finish
echo "Waiting for haplo job to complete..."
while qstat -j $HAPLO_JOB_ID >/dev/null 2>&1; do
    sleep 10
    echo "Still waiting for haplo job to complete..."
done
echo "Haplo job has completed, collecting resource usage..."

# Get resource usage - using more precise grep patterns
maxvmem=\$(qacct -j $HAPLO_JOB_ID | grep "^maxvmem" | awk '{print \$2}' | sed 's/G//')
maxrss=\$(qacct -j $HAPLO_JOB_ID | grep "^maxrss" | awk '{print \$2}' | sed 's/M//')
grid_mem=\$(qacct -j $HAPLO_JOB_ID | grep "^mem" | awk '{print \$2}')

# Debug output
echo "Extracted values: maxvmem=\$maxvmem, maxrss=\$maxrss, grid_mem=\$grid_mem"

# Update stats file
sed -i "s/NA\\tNA\\tNA\$/\\\$maxvmem\\t\\\$maxrss\\t\\\$grid_mem/" stats/hgdp_haplo_stats.tsv
echo "Stats file updated with resource usage information"
echo "====================================================="
EOF

chmod +x collect_stats.sh

# Submit the collection script as a separate job
STATS_JOB_ID=$(qsub -N "collect_stats_${chr}" -v HAPLO_JOB_ID=$HAPLO_JOB_ID collect_stats.sh)
echo "Stats collection job submitted with ID: $STATS_JOB_ID"
echo "=====================================================" 