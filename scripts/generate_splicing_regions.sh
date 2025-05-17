#!/bin/bash

# Input exon BED file (must be 6-column: chrom, start, end, name, score, strand)
EXON_BED="gencodev47_basic_coding_exon.bed"

# Output file names based on short names
REGIONS=("5ss_can" "5ss_iprox" "5ss_eprox" "3ss_can" "3ss_iprox" "3ss_eprox" "bp_region" "exon_core")

# Clear output files
for region in "${REGIONS[@]}"; do
    > "${region}.bed"
done

# Generate all regions
awk 'BEGIN {OFS="\t"}
{
    chrom = $1;
    start = $2;
    end   = $3;
    name  = $4;
    score = $5;
    strand = $6;

    # 5′ss: donor site
    if (strand == "+") {
        # Canonical 5′ss [+1, +2]
        print chrom, end, end+2, "5ss_can", score, strand >> "5ss_can.bed";

        # Intronic proximal 5′ss [+3, +6]
        print chrom, end+2, end+6, "5ss_iprox", score, strand >> "5ss_iprox.bed";

        # Exonic proximal 5′ss [-3, 0]
        print chrom, end-3, end, "5ss_eprox", score, strand >> "5ss_eprox.bed";
    } else {
        # Canonical 5′ss [-2, -1]
        print chrom, start-2, start, "5ss_can", score, strand >> "5ss_can.bed";

        # Intronic proximal 5′ss [-6, -3]
        print chrom, start-6, start-2, "5ss_iprox", score, strand >> "5ss_iprox.bed";

        # Exonic proximal 5′ss [0, +3]
        print chrom, start, start+3, "5ss_eprox", score, strand >> "5ss_eprox.bed";
    }

    # 3′ss: acceptor site
    if (strand == "+") {
        # Canonical 3′ss [-2, -1]
        print chrom, start-2, start, "3ss_can", score, strand >> "3ss_can.bed";

        # Intronic proximal 3′ss [-17, -3]
        print chrom, start-17, start-2, "3ss_iprox", score, strand >> "3ss_iprox.bed";

        # Branchpoint [-40, -18]
        print chrom, start-40, start-17, "bp_region", score, strand >> "bp_region.bed";

        # Exonic proximal 3′ss [0, +3]
        print chrom, start, start+3, "3ss_eprox", score, strand >> "3ss_eprox.bed";
    } else {
        # Canonical 3′ss [+1, +2]
        print chrom, end, end+2, "3ss_can", score, strand >> "3ss_can.bed";

        # Intronic proximal 3′ss [+3, +17]
        print chrom, end+2, end+17, "3ss_iprox", score, strand >> "3ss_iprox.bed";

        # Branchpoint [+18, +40]
        print chrom, end+17, end+40, "bp_region", score, strand >> "bp_region.bed";

        # Exonic proximal 3′ss [-3, 0]
        print chrom, end-3, end, "3ss_eprox", score, strand >> "3ss_eprox.bed";
    }

    # Exon body excluding splice sites: [start+3, end-3] for +, [start+3, end-3] for -
    core_start = (strand == "+") ? start + 3 : start + 3;
    core_end   = (strand == "+") ? end - 3 : end - 3;

    if (core_start < core_end) {
        print chrom, core_start, core_end, "exon_core", score, strand >> "exon_core.bed";
    }
}' "$EXON_BED"

echo "Region BEDs written:"
for region in "${REGIONS[@]}"; do
    echo "  • ${region}.bed"
done
