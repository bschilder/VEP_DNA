"""
Function to create a linear diagram showing genomic regions around an exon with splice sites.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_splice_site_regions(figsize=(14, 3), exon_core_length=50, intron_distal_length=30):
    """
    Create a linear diagram showing genomic regions around an exon with splice sites.
    
    Parameters
    ----------
    figsize : tuple
        Figure size (width, height)
    exon_core_length : int
        Length of the exonic core region (in arbitrary units for visualization)
    intron_distal_length : int
        Length of the intronic distal regions (in arbitrary units for visualization)
    """
    # Define region lengths (in bp)
    region_lengths = {
        'intron_distal_left': intron_distal_length,
        'intron_proximal_left': 33,
        '3ss_iprox': 15,
        '3ss_can': 2,
        '3ss_eprox': 3,
        'exon_core': exon_core_length,
        '5ss_eprox': 3,
        '5ss_can': 4,
        '5ss_iprox': 4,
        'intron_proximal_right': 44,
        'intron_distal_right': intron_distal_length,
    }
    
    # Calculate cumulative positions
    positions = [0]
    for region in region_lengths.values():
        positions.append(positions[-1] + region)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set y position for the main line
    y_pos = 0.5
    
    # Draw one continuous gray horizontal line for the entire sequence
    intron_color = 'gray'
    ax.plot([positions[0], positions[-1]], [y_pos, y_pos], 
            color=intron_color, linewidth=3, solid_capstyle='round', zorder=1)
    
    # Draw exon box (thicker gray rectangle) on top of the line
    exon_start = positions[4]  # Start of 3'ss exonic proximal
    exon_end = positions[7]  # End of 5'ss exonic proximal
    exon_height = 0.2  # Make it thicker to be visually distinct
    exon_rect = patches.Rectangle((exon_start, y_pos - exon_height/2), 
                                  exon_end - exon_start, exon_height,
                                  facecolor='gray', edgecolor='gray', linewidth=0,
                                  zorder=2)
    ax.add_patch(exon_rect)
    
    # Add vertical dotted lines to separate regions
    # Draw lines at all boundaries except the very first and last
    for i in range(1, len(positions) - 1):
        ax.axvline(positions[i], color='black', linestyle=':', linewidth=1, 
                   alpha=0.6, zorder=3)
    
    # Add region labels below the line
    label_y = y_pos - 0.25
    region_names = [
        ('Intronic distal', positions[0], positions[1], None),
        ('Intronic proximal', positions[1], positions[2], '33 bp'),
        ("3'ss intronic proximal", positions[2], positions[3], '15 bp'),
        ("3'ss canonical", positions[3], positions[4], '2 bp'),
        ("3'ss exonic proximal", positions[4], positions[5], '3 bp'),
        ('Exonic core', positions[5], positions[6], None),
        ("5'ss exonic proximal", positions[6], positions[7], '3 bp'),
        ("5'ss canonical", positions[7], positions[8], '4 bp'),
        ("5'ss intronic proximal", positions[8], positions[9], '4 bp'),
        ('Intronic proximal', positions[9], positions[10], '44 bp'),
        ('Intronic distal', positions[10], positions[11], None),
    ]
    
    for name, start, end, length in region_names:
        center = (start + end) / 2
        if length:
            label_text = f"{name}\n({length})"
        else:
            label_text = name
        ax.text(center, label_y, label_text, ha='center', va='top', 
                fontsize=9, fontweight='normal')
    
    # Add splice site labels above the boundaries
    # 3'ss label - at boundary between 3'ss canonical (intron) and 3'ss exonic proximal (exon)
    ss3_pos = positions[4]  # Boundary between 3'ss canonical and 3'ss exonic proximal
    ax.text(ss3_pos, y_pos + 0.22, "3'ss", ha='center', va='bottom',
            fontsize=11, fontweight='bold', zorder=4)
    # Draw a line from label to main horizontal line
    ax.plot([ss3_pos, ss3_pos], [y_pos + 0.18, y_pos], 
            color='black', linewidth=1.5, linestyle='-', zorder=4)
    
    # 5'ss label - at boundary between 5'ss exonic proximal (exon) and 5'ss canonical (intron)
    ss5_pos = positions[7]  # Boundary between 5'ss exonic proximal and 5'ss canonical
    ax.text(ss5_pos, y_pos + 0.22, "5'ss", ha='center', va='bottom',
            fontsize=11, fontweight='bold', zorder=4)
    # Draw a line from label to main horizontal line
    ax.plot([ss5_pos, ss5_pos], [y_pos + 0.18, y_pos], 
            color='black', linewidth=1.5, linestyle='-', zorder=4)
    
    # Set axis properties
    ax.set_xlim(positions[0] - 5, positions[-1] + 5)
    ax.set_ylim(-0.5, 0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # Create the diagram
    fig, ax = plot_splice_site_regions()
    plt.savefig('splice_site_regions_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

