# Publication-Ready Figure Generation

This guide explains how to generate high-quality figures suitable for IEEE paper submissions.

## Overview

The `generate_paper_figures.py` script creates publication-ready visualizations in multiple formats (PNG, PDF, SVG) with proper academic styling.

## Usage

### Basic Usage

```bash
python src/generate_paper_figures.py
```

### Custom Paths

```bash
python src/generate_paper_figures.py \
    --project-root /path/to/project \
    --output-dir paper_figures
```

## Generated Figures

The script generates the following figures:

### 1. Model Comparison (`figure_model_comparison.*`)
- Bar chart comparing all models
- 95% confidence intervals
- Error bars showing standard deviation
- Suitable for: Results section, comparison tables

### 2. K-Fold Distribution (`figure_kfold_distribution.*`)
- Box plots showing distribution across folds
- Mean and quartiles clearly marked
- Suitable for: Methodology section, robustness analysis

### 3. Metrics Comparison (`figure_metrics_comparison.*`)
- Grouped bar chart with multiple metrics
- Accuracy, Precision, Recall, F1 scores
- Suitable for: Comprehensive results comparison

### 4. Statistical Significance (`figure_statistical_significance.*`)
- Heatmap of p-values from pairwise t-tests
- Color-coded significance levels
- Suitable for: Statistical analysis section

### 5. Summary Table (`table_results_summary.*`)
- CSV and LaTeX formats
- Complete statistics for all models
- Ready for direct inclusion in LaTeX documents

## Figure Specifications

All figures are generated with:
- **Resolution**: 300 DPI (suitable for print)
- **Font**: Times New Roman (serif, academic standard)
- **Format**: PNG, PDF, SVG (multiple formats for flexibility)
- **Style**: Publication-quality with proper labels and legends

## LaTeX Integration

### Including Figures

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figure_model_comparison.pdf}
    \caption{Model performance comparison with 95\% confidence intervals.}
    \label{fig:model_comparison}
\end{figure}
```

### Including Tables

```latex
\input{table_results_summary.tex}
```

## Customization

### Changing Figure Style

Edit `src/generate_paper_figures.py` and modify the matplotlib parameters:

```python
matplotlib.rcParams.update({
    'font.size': 11,           # Adjust font size
    'font.family': 'serif',     # Change font family
    'figure.dpi': 300,          # Adjust resolution
    # ... more parameters
})
```

### Adding Custom Figures

Extend the script by adding new plotting functions following the same pattern:

```python
def plot_custom_figure(results: Dict, output_dir: Path):
    """Your custom figure generation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... your plotting code ...
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(output_dir / f'figure_custom.{fmt}', format=fmt, dpi=300)
    plt.close()
```

## Best Practices for IEEE Papers

1. **Figure Size**: Use appropriate sizes (typically 3.5" width for single column, 7" for double column)
2. **Font Size**: Ensure text is readable when scaled (minimum 8pt)
3. **Color**: Use colorblind-friendly palettes (provided by seaborn)
4. **Labels**: Always include axis labels, units, and legends
5. **Captions**: Write descriptive captions explaining what the figure shows

## Troubleshooting

### Missing Results

If no figures are generated:
- Ensure Stage 5 training has completed
- Check that `data/training_results/` contains model directories
- Verify `fold_results.csv` files exist

### Font Issues

If fonts don't render correctly:
- Install Times New Roman on your system
- Or change to a system font in the matplotlib parameters
- PDF fonts require TrueType fonts (already configured)

### Resolution Issues

For higher resolution:
- Change `dpi=300` to `dpi=600` in save commands
- Note: Higher DPI increases file size significantly

## Example Workflow

1. **Complete Training**:
   ```bash
   python src/run_new_pipeline.py --only-stage 5
   ```

2. **Generate Figures**:
   ```bash
   python src/generate_paper_figures.py
   ```

3. **Review Figures**:
   - Check `paper_figures/` directory
   - Verify all figures are generated correctly
   - Review figure quality and labels

4. **Include in Paper**:
   - Copy PDF files to your LaTeX document directory
   - Include using `\includegraphics`
   - Add appropriate captions and labels

## Figure Checklist

Before submitting your paper, verify:

- [ ] All figures have clear, readable labels
- [ ] Legends are present and informative
- [ ] Error bars/confidence intervals are shown where appropriate
- [ ] Figures are high resolution (300+ DPI)
- [ ] Color schemes are colorblind-friendly
- [ ] Figures match IEEE formatting guidelines
- [ ] Captions accurately describe the content
- [ ] All figures are referenced in the text

