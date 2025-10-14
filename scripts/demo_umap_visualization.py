#!/usr/bin/env python3
"""
Demo script for Beautiful UMAP Visualization with Bokeh

This script creates sample UMAP data and demonstrates the beautiful visualization
without requiring the full VEP dataset.
"""

import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import viridis, inferno
import warnings

warnings.filterwarnings('ignore')

def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration."""
    print("Generating sample UMAP data...")
    
    # Generate random high-dimensional data with some structure
    np.random.seed(42)
    n_features = 50
    
    # Create clusters for different super populations
    super_pops = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
    n_per_pop = n_samples // len(super_pops)
    
    data = []
    labels = []
    sample_names = []
    
    for i, pop in enumerate(super_pops):
        # Create cluster center
        center = np.random.normal(0, 2, n_features)
        
        # Generate samples around this center
        pop_data = np.random.normal(center, 1, (n_per_pop, n_features))
        data.append(pop_data)
        labels.extend([pop] * n_per_pop)
        sample_names.extend([f"{pop}_sample_{j}" for j in range(n_per_pop)])
    
    # Add some reference samples
    ref_samples = np.random.normal(0, 1.5, (n_samples % len(super_pops), n_features))
    data.append(ref_samples)
    labels.extend(['REF'] * len(ref_samples))
    sample_names.extend([f"REF_sample_{j}" for j in range(len(ref_samples))])
    
    # Combine all data
    X = np.vstack(data)
    
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Super populations: {set(labels)}")
    
    return X, labels, sample_names

def run_umap_demo(X, labels, sample_names):
    """Run UMAP on sample data."""
    print("Running UMAP...")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run UMAP
    dr_out = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='cosine'
    ).fit_transform(X_scaled)
    
    # Create DataFrame
    umap_df = pd.DataFrame({
        'sample': sample_names,
        'UMAP1': dr_out[:, 0],
        'UMAP2': dr_out[:, 1],
        'Super Population': labels
    })
    
    print(f"UMAP completed. Shape: {umap_df.shape}")
    print(umap_df['Super Population'].value_counts())
    
    return umap_df

def create_density_landscape_demo(umap_df):
    """Create 2D density landscape for demo."""
    print("Creating 2D density landscape...")
    
    # Define grid for density calculation
    x_min, x_max = umap_df['UMAP1'].min() - 1, umap_df['UMAP1'].max() + 1
    y_min, y_max = umap_df['UMAP2'].min() - 1, umap_df['UMAP2'].max() + 1
    
    # Create meshgrid
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Calculate kernel density estimation
    values = np.vstack([umap_df['UMAP1'], umap_df['UMAP2']])
    kernel = gaussian_kde(values, bw_method=0.3)
    density = np.reshape(kernel(positions).T, xx.shape)
    
    # Normalize density for better visualization
    density = (density - density.min()) / (density.max() - density.min())
    
    print(f"Density landscape created with shape: {density.shape}")
    
    return density, xx, yy, x_min, x_max, y_min, y_max

def create_demo_plot(umap_df, density, xx, yy, x_min, x_max, y_min, y_max, 
                    output_file_name="demo_umap_landscape.html", dark_theme=False):
    """Create beautiful demo plot."""
    print("Creating beautiful demo visualization...")
    
    # Define beautiful color palette for super populations
    superpop_colors = {
        'AFR': '#E64B35',  # Deep red
        'AMR': '#4DBBD5',  # Sky blue
        'EAS': '#00A087',  # Teal
        'EUR': '#3C5488',  # Navy blue
        'SAS': '#F39B7F',  # Coral
        'REF': '#8491B4'   # Gray
    }
    
    # Choose theme colors
    if dark_theme:
        bg_color = "#0a0a0a"
        border_color = "#0a0a0a"
        title_color = "#ffffff"
        axis_color = "#ffffff"
        label_color = "#cccccc"
        grid_color = "#333333"
        legend_bg = "#1a1a1a"
        legend_border = "#444444"
        palette = inferno(256)
        title = "Demo: Genetic Landscape Visualization"
    else:
        bg_color = "#f8f9fa"
        border_color = "#ffffff"
        title_color = "#2c3e50"
        axis_color = "#2c3e50"
        label_color = "#2c3e50"
        grid_color = "#ecf0f1"
        legend_bg = "#ffffff"
        legend_border = "#bdc3c7"
        palette = viridis(256)
        title = "Demo: UMAP Landscape of Genetic Variation"
    
    # Create the main figure
    p = figure(
        width=1200 if dark_theme else 1000,
        height=900 if dark_theme else 800,
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color=bg_color,
        border_fill_color=border_color
    )
    
    # Add density landscape as background
    if dark_theme:
        density_enhanced = np.power(density, 0.7)  # Enhance contrast for dark theme
    else:
        density_enhanced = density
    
    p.image(
        image=[density_enhanced],
        x=x_min,
        y=y_min,
        dw=x_max-x_min,
        dh=y_max-y_min,
        palette=palette,
        alpha=0.4 if dark_theme else 0.3,
        level="underlay"
    )
    
    # Add contour lines for landscape effect (simplified approach)
    density_smooth = gaussian_filter(density, sigma=1)
    
    # Create simple contour lines manually
    levels = np.linspace(density_smooth.min(), density_smooth.max(), 8)
    contour_color = '#ffffff' if dark_theme else '#2c3e50'
    
    # Add some simple contour-like lines for landscape effect
    for level in levels[1:-1]:  # Skip first and last levels
        # Find points near this level
        mask = np.abs(density_smooth - level) < 0.05
        if np.any(mask):
            # Get coordinates where mask is True
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 10:  # Only draw if we have enough points
                # Convert to actual coordinates
                x_actual = x_min + (x_max - x_min) * x_coords / 100
                y_actual = y_min + (y_max - y_min) * y_coords / 100
                
                # Draw lines connecting nearby points
                for i in range(0, len(x_actual), 5):
                    if i + 1 < len(x_actual):
                        p.line(
                            [x_actual[i], x_actual[i+1]],
                            [y_actual[i], y_actual[i+1]],
                            line_color=contour_color,
                            line_alpha=0.05,
                            line_width=0.5,
                            level="underlay"
                        )
    
    # Add scatter points for each super population
    for superpop in umap_df['Super Population'].unique():
        pop_data = umap_df[umap_df['Super Population'] == superpop]
        
        # Create ColumnDataSource
        source = ColumnDataSource(data=dict(
            x=pop_data['UMAP1'],
            y=pop_data['UMAP2'],
            sample=pop_data['sample'],
            superpop=pop_data['Super Population']
        ))
        
        color = superpop_colors.get(superpop, '#95a5a6')
        
        # Add glow effect for dark theme
        if dark_theme:
            p.scatter(
                'x', 'y',
                source=source,
                size=15,
                fill_color=color,
                line_color=color,
                line_width=2,
                alpha=0.2,
                name=f"{superpop}_glow"
            )
        
        # Add main points
        scatter = p.scatter(
            'x', 'y',
            source=source,
            size=8 if not dark_theme else 6,
            fill_color=color,
            line_color='white',
            line_width=1.5,
            alpha=0.9,
            legend_label=superpop,
            name=superpop
        )
        
        # Add hover tool
        hover = HoverTool(
            renderers=[scatter],
            tooltips=[
                ('Sample', '@sample'),
                ('Super Population', '@superpop'),
                ('UMAP1', '@x{0.000}'),
                ('UMAP2', '@y{0.000}')
            ]
        )
        p.add_tools(hover)
    
    # Style the plot
    p.title.text_font_size = '24pt' if dark_theme else '20pt'
    p.title.text_font_style = 'bold'
    p.title.text_color = title_color
    
    # Style axes
    p.xaxis.axis_label = 'UMAP Dimension 1'
    p.yaxis.axis_label = 'UMAP Dimension 2'
    p.xaxis.axis_label_text_font_size = '16pt' if dark_theme else '14pt'
    p.yaxis.axis_label_text_font_size = '16pt' if dark_theme else '14pt'
    p.xaxis.major_label_text_font_size = '14pt' if dark_theme else '12pt'
    p.yaxis.major_label_text_font_size = '14pt' if dark_theme else '12pt'
    p.xaxis.axis_label_text_color = axis_color
    p.yaxis.axis_label_text_color = axis_color
    p.xaxis.major_label_text_color = label_color
    p.yaxis.major_label_text_color = label_color
    
    # Style grid
    p.grid.grid_line_color = grid_color
    p.grid.grid_line_alpha = 0.3
    
    # Style legend
    p.legend.title = 'Super Population'
    p.legend.title_text_font_size = '16pt' if dark_theme else '14pt'
    p.legend.label_text_font_size = '14pt' if dark_theme else '12pt'
    p.legend.location = 'top_right'
    p.legend.background_fill_alpha = 0.8
    p.legend.background_fill_color = legend_bg
    p.legend.border_line_color = legend_border
    p.legend.title_text_color = title_color
    p.legend.label_text_color = title_color
    
    # Save the plot
    output_file(output_file_name, title=title)
    save(p)
    print(f"Demo plot saved as '{output_file_name}'")
    
    return p

def main():
    """Main function for demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create demo UMAP visualization with Bokeh')
    parser.add_argument('--dark', action='store_true', help='Use dark theme')
    parser.add_argument('--output', type=str, default='demo_umap_landscape.html', 
                       help='Output file name')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    print("=== Demo: Beautiful UMAP Visualization with Bokeh ===")
    print(f"Theme: {'Dark' if args.dark else 'Light'}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.samples}")
    print()
    
    # Generate sample data
    X, labels, sample_names = generate_sample_data(args.samples)
    
    # Run UMAP
    umap_df = run_umap_demo(X, labels, sample_names)
    
    # Create density landscape
    density, xx, yy, x_min, x_max, y_min, y_max = create_density_landscape_demo(umap_df)
    
    # Create beautiful plot
    p = create_demo_plot(umap_df, density, xx, yy, x_min, x_max, y_min, y_max,
                        args.output, args.dark)
    
    # Print summary statistics
    print("\n=== Demo UMAP Landscape Summary ===")
    print(f"Total samples: {len(umap_df)}")
    print("\nSuper Population distribution:")
    print(umap_df['Super Population'].value_counts())
    print(f"\nUMAP1 range: [{umap_df['UMAP1'].min():.3f}, {umap_df['UMAP1'].max():.3f}]")
    print(f"UMAP2 range: [{umap_df['UMAP2'].min():.3f}, {umap_df['UMAP2'].max():.3f}]")
    print("\nDensity landscape statistics:")
    print(f"Min density: {density.min():.6f}")
    print(f"Max density: {density.max():.6f}")
    print(f"Mean density: {density.mean():.6f}")
    
    print(f"\n✅ Demo UMAP visualization completed!")
    print(f"📁 Output file: {args.output}")
    print(f"🌐 Open the HTML file in your browser to view the interactive plot")
    print(f"💡 This is a demo with synthetic data. Use the main script for real VEP data.")

if __name__ == "__main__":
    main() 