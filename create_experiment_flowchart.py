#!/usr/bin/env python3
"""
Generuje profesjonalny schemat blokowy Eksperymentu 1
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def create_flowchart():
    """Tworzy schemat blokowy procesu eksperymentalnego"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Kolory (profesjonalne, przyjazne dla daltonistów)
    color_data = '#E8F4F8'       # Jasny niebieski - dane
    color_process = '#FFE5CC'    # Jasny pomarańczowy - procesy
    color_method = '#E8F5E9'     # Jasny zielony - metody
    color_eval = '#F3E5F5'       # Jasny fiolet - ewaluacja
    
    box_style = "round,pad=0.1"
    
    # Funkcja pomocnicza do rysowania boxów
    def draw_box(x, y, width, height, text, color, fontsize=10, fontweight='normal'):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle=box_style,
            edgecolor='black',
            facecolor=color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight,
                wrap=True, zorder=3)
    
    # Funkcja do rysowania strzałek
    def draw_arrow(x1, y1, x2, y2, label='', style='->'):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            linewidth=2,
            color='black',
            zorder=1,
            mutation_scale=20
        )
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mid_x + 0.3, mid_y, label,
                   fontsize=8, style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
    
    # Tytuł
    ax.text(5, 13.5, 'Experiment 1: Evaluation Pipeline', 
            ha='center', fontsize=16, fontweight='bold')
    
    # ========== STAGE 1: DATA PREPARATION ==========
    y_start = 12.5
    ax.text(0.5, y_start + 0.3, 'Stage 1:', fontsize=11, fontweight='bold')
    draw_box(0.5, y_start - 0.7, 4, 0.8, 
             'Original Industrial\nTime Series Datasets\n(Boiler, Pump, Vibration)',
             color_data, fontsize=9, fontweight='bold')
    
    # Arrow down
    draw_arrow(2.5, y_start - 0.7, 2.5, y_start - 1.5)
    
    # ========== STAGE 2: MISSING DATA INJECTION ==========
    y_stage2 = y_start - 2.0
    ax.text(0.5, y_stage2 + 0.8, 'Stage 2:', fontsize=11, fontweight='bold')
    
    # Missing data injection box
    draw_box(0.5, y_stage2, 4, 1.2,
             'Missing Data Injection\n\n3 Mechanisms: MCAR, MAR, MNAR\n3 Rates: 2%, 5%, 10%\n10 Iterations each',
             color_process, fontsize=9, fontweight='bold')
    
    # Result
    draw_arrow(2.5, y_stage2, 2.5, y_stage2 - 0.8)
    draw_box(0.5, y_stage2 - 1.5, 4, 0.7,
             'Corrupted Series + Binary Mask\n(270 configs/dataset)',
             color_data, fontsize=9)
    
    # ========== STAGE 3: RECONSTRUCTION (SPLIT) ==========
    y_stage3 = y_stage2 - 2.5
    ax.text(0.5, y_stage3 + 0.5, 'Stage 3:', fontsize=11, fontweight='bold')
    
    # Split arrow
    draw_arrow(2.5, y_stage2 - 1.5, 2.5, y_stage3 + 0.3)
    draw_arrow(2.5, y_stage3 + 0.3, 0.8, y_stage3)
    draw_arrow(2.5, y_stage3 + 0.3, 4.2, y_stage3)
    
    # LEFT BRANCH: Classical Methods
    draw_box(0.2, y_stage3 - 1.5, 2, 1.5,
             'Classical Methods\n(Time Domain)\n\n• Statistical: mean, median\n• Interpolation: linear, cubic, spline\n• ML: k-NN, SARIMAX',
             color_method, fontsize=8)
    
    # RIGHT BRANCH: Image-based Methods
    draw_box(2.7, y_stage3 - 1.5, 2.1, 1.5,
             'Image-based Methods\n\n1. Transform to image\n   (GAF/MTF/RP/SPEC)\n2. Inpaint (U-Net/SD2)\n3. Inverse transform',
             color_method, fontsize=8)
    
    # Image examples (small boxes)
    img_y = y_stage3 - 2.2
    img_size = 0.35
    img_labels = ['GAF', 'MTF', 'RP', 'SPEC']
    for i, label in enumerate(img_labels):
        x_pos = 5.3 + i * 0.45
        rect = Rectangle((x_pos, img_y), img_size, img_size,
                        facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos + img_size/2, img_y - 0.15, label,
               fontsize=6, ha='center')
    
    # Arrows down from both branches
    draw_arrow(1.2, y_stage3 - 1.5, 1.2, y_stage3 - 2.8)
    draw_arrow(3.75, y_stage3 - 1.5, 3.75, y_stage3 - 2.8)
    
    # Merge
    y_merge = y_stage3 - 3.2
    draw_arrow(1.2, y_stage3 - 2.8, 2.5, y_merge)
    draw_arrow(3.75, y_stage3 - 2.8, 2.5, y_merge)
    
    draw_box(0.5, y_merge - 0.7, 4, 0.7,
             'Reconstructed Time Series',
             color_data, fontsize=10, fontweight='bold')
    
    # ========== STAGE 4: EVALUATION ==========
    y_stage4 = y_merge - 1.5
    ax.text(0.5, y_stage4 + 0.2, 'Stage 4:', fontsize=11, fontweight='bold')
    draw_arrow(2.5, y_merge - 0.7, 2.5, y_stage4 + 0.2)
    
    # Evaluation box
    draw_box(0.5, y_stage4 - 1.2, 4, 1.2,
             'Evaluation (Masked Regions Only)\n\n1. Extract masked segments\n2. Compute |Original - Reconstructed|\n3. Average → MAD',
             color_eval, fontsize=9, fontweight='bold')
    
    # Final metric
    draw_arrow(2.5, y_stage4 - 1.2, 2.5, y_stage4 - 2.0)
    draw_box(0.5, y_stage4 - 2.5, 4, 0.5,
             'Mean Absolute Difference (MAD)\n↓ Lower = Better',
             color_eval, fontsize=10, fontweight='bold')
    
    # ========== RIGHT SIDE: LEGEND & INFO ==========
    legend_x = 5.5
    legend_y = 12
    
    ax.text(legend_x, legend_y, 'Pipeline Components:', 
            fontsize=11, fontweight='bold')
    
    # Color legend
    colors_info = [
        (color_data, 'Data'),
        (color_process, 'Processing'),
        (color_method, 'Methods'),
        (color_eval, 'Evaluation')
    ]
    
    for i, (color, label) in enumerate(colors_info):
        y_pos = legend_y - 0.5 - i * 0.4
        rect = Rectangle((legend_x, y_pos - 0.15), 0.3, 0.25,
                        facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.4, y_pos, label, fontsize=9, va='center')
    
    # Key numbers box
    info_y = 9
    ax.text(legend_x, info_y + 0.3, 'Experimental Scale:', 
            fontsize=10, fontweight='bold')
    
    info_text = [
        'Datasets: 3',
        'Mechanisms: 3',
        'Missing rates: 3',
        'Iterations: 10',
        '─────────────',
        'Configs/dataset: 270',
        'Total experiments: 810',
        '',
        'Classical methods: 15',
        'Image methods: 16',
        'Total methods: 31'
    ]
    
    box_height = len(info_text) * 0.25 + 0.3
    draw_box(legend_x, info_y - box_height, 2.8, box_height,
             '\n'.join(info_text), 'lightyellow', fontsize=8)
    
    # Timeline indicator
    timeline_y = 4.5
    ax.text(legend_x, timeline_y + 0.3, 'Timeline:', 
            fontsize=10, fontweight='bold')
    
    timeline_items = [
        ('1', 'Load datasets', 0.5),
        ('2', 'Inject missing data', 0.5),
        ('3', 'Reconstruct (15-30s/method)', 1.5),
        ('4', 'Evaluate (instant)', 0.5)
    ]
    
    curr_y = timeline_y
    for num, desc, _ in timeline_items:
        curr_y -= 0.35
        ax.text(legend_x, curr_y, f'{num}. {desc}', fontsize=8)
    
    # Note about evaluation
    note_y = 1.5
    note_text = 'Note: Evaluation only on\nmasked (missing) regions,\nnot observed values'
    draw_box(legend_x, note_y - 0.8, 2.8, 0.8,
             note_text, '#FFF9C4', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_simple_flowchart():
    """Tworzy uproszczoną wersję schematu - bardziej kompaktową"""
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Kolory
    color_input = '#E3F2FD'
    color_process = '#FFF3E0'
    color_output = '#E8F5E9'
    
    def draw_box(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                               linewidth=2, color='black', mutation_scale=15)
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 9.5, 'Experiment 1: Simplified Pipeline', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Flow (przesunięte w lewo, żeby zrobić miejsce na prawo)
    draw_box(1, 8, 5, 0.8, 'Original Time Series\n(3 datasets)', color_input)
    draw_arrow(3.5, 8, 3.5, 7.2)
    
    draw_box(1, 6.2, 5, 1, 'Inject Missing Data\nMCAR/MAR/MNAR @ 2%/5%/10%\n10 iterations', color_process)
    draw_arrow(3.5, 6.2, 3.5, 5.4)
    
    draw_box(1, 4.6, 5, 0.8, 'Corrupted Series + Mask', color_input)
    
    # Split
    draw_arrow(3.5, 4.6, 2, 3.8)
    draw_arrow(3.5, 4.6, 5, 3.8)
    
    draw_box(0.2, 2.8, 3.5, 1, 'Classical\nReconstruction\n(15 methods)', color_process, fontsize=8)
    draw_box(4, 2.8, 3.5, 1, 'Image-based\nReconstruction\n(16 methods)', color_process, fontsize=8)
    
    # Merge
    draw_arrow(2, 2.8, 3.5, 2)
    draw_arrow(5.7, 2.8, 3.5, 2)
    
    draw_box(1, 1.2, 5, 0.8, 'Reconstructed Series', color_output)
    draw_arrow(3.5, 1.2, 3.5, 0.4)
    
    draw_box(1, -0.4, 5, 0.8, 'MAD Metric\n(on masked regions)', color_output)
    
    # EXPERIMENTAL SCALE BOX (po prawej stronie, dobrze widoczny)
    scale_x = 8
    scale_y = 6.5
    
    # Nagłówek
    ax.text(scale_x, scale_y + 0.5, 'Experimental Scale:', 
            fontsize=11, fontweight='bold', ha='left')
    
    # Box z informacjami
    scale_info = [
        'Datasets: 3',
        'Mechanisms: 3',
        'Missing rates: 3',
        'Iterations: 10',
        '─────────────',
        'Configs/dataset: 270',
        'Total configs: 810',
        '',
        'Classical methods: 15',
        'Image methods: 16',
        'Total methods: 31'
    ]
    
    info_box = FancyBboxPatch(
        (scale_x, scale_y - 3), 3.5, 3.2,
        boxstyle="round,pad=0.15",
        edgecolor='darkblue',
        facecolor='#FFF9C4',
        linewidth=2
    )
    ax.add_patch(info_box)
    
    # Tekst wewnątrz boxa
    for i, line in enumerate(scale_info):
        y_pos = scale_y + 2.7 - i * 0.27
        if '─' in line:
            ax.text(scale_x + 1.75, y_pos, line, 
                   fontsize=8, ha='center', color='gray')
        else:
            ax.text(scale_x + 0.2, y_pos, line, 
                   fontsize=8, ha='left', va='center')
    
    # Note box (pod spodem)
    note_y = 2
    note_box = FancyBboxPatch(
        (scale_x, note_y - 0.8), 3.5, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='darkgreen',
        facecolor='#E8F5E9',
        linewidth=1.5
    )
    ax.add_patch(note_box)
    
    ax.text(scale_x + 1.75, note_y - 0.4, 
           'Evaluation only on\nmasked (missing) regions', 
           fontsize=8, ha='center', va='center', style='italic')
    
    plt.tight_layout()
    return fig

def create_latex_tikz():
    """Generuje kod LaTeX z TikZ dla profesjonalnego schematu"""
    
    tikz_code = r'''\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.2cm,
    box/.style={rectangle, draw=black, thick, fill=blue!10, 
                rounded corners, minimum width=3cm, minimum height=0.8cm,
                text width=2.8cm, align=center, font=\small},
    process/.style={rectangle, draw=black, thick, fill=orange!10, 
                   rounded corners, minimum width=3cm, minimum height=0.8cm,
                   text width=2.8cm, align=center, font=\small},
    method/.style={rectangle, draw=black, thick, fill=green!10, 
                  rounded corners, minimum width=2.5cm, minimum height=1.2cm,
                  text width=2.3cm, align=center, font=\footnotesize},
    arrow/.style={->, >=stealth, thick}
]

% Stage 1: Original Data
\node[box] (data) at (0,0) {\textbf{Original Time Series}\\3 Industrial Datasets};

% Stage 2: Missing Data Injection
\node[process, below=of data] (inject) {\textbf{Missing Data Injection}\\
    MCAR, MAR, MNAR\\
    2\%, 5\%, 10\%\\
    10 iterations};

\node[box, below=of inject] (corrupted) {\textbf{Corrupted Series}\\+ Binary Mask};

% Stage 3: Reconstruction (split)
\node[method, below left=1.5cm and 2cm of corrupted] (classical) {
    \textbf{Classical Methods}\\[0.1cm]
    • Statistical\\
    • Interpolation\\
    • k-NN, SARIMAX
};

\node[method, below right=1.5cm and 2cm of corrupted] (image) {
    \textbf{Image-based}\\[0.1cm]
    1. Transform\\
    2. Inpaint\\
    3. Inv. Transform
};

% Stage 4: Reconstructed
\node[box, below=3cm of corrupted] (recon) {\textbf{Reconstructed Series}};

% Stage 5: Evaluation
\node[process, below=of recon] (eval) {\textbf{Evaluation}\\
    Extract masked segments\\
    Compute MAD};

\node[box, below=of eval, fill=purple!10] (mad) {\textbf{MAD Metric}\\
    Lower = Better};

% Arrows
\draw[arrow] (data) -- (inject);
\draw[arrow] (inject) -- (corrupted);
\draw[arrow] (corrupted) -| (classical);
\draw[arrow] (corrupted) -| (image);
\draw[arrow] (classical) |- (recon);
\draw[arrow] (image) |- (recon);
\draw[arrow] (recon) -- (eval);
\draw[arrow] (eval) -- (mad);

% Annotations
\node[right=3cm of inject, text width=3cm, font=\footnotesize] (info) {
    \textbf{Scale:}\\
    270 configs/dataset\\
    31 methods\\
    8,370 experiments
};

\draw[dashed, gray] (inject.east) -- (info.west);

\end{tikzpicture}
\caption{Experimental pipeline for Experiment~1. Time series are systematically corrupted, reconstructed using classical or image-based methods, and evaluated using MAD metric computed on masked (missing) regions only.}
\label{fig:exp1_pipeline}
\end{figure}'''
    
    return tikz_code

def main():
    """Generuje wszystkie wizualizacje"""
    
    print("Generating flowcharts...")
    
    # Detailed flowchart
    print("1. Creating detailed flowchart...")
    fig1 = create_flowchart()
    fig1.savefig('reports/experiment_1_flowchart_detailed.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/experiment_1_flowchart_detailed.png")
    
    # Simple flowchart
    print("2. Creating simple flowchart...")
    fig2 = create_simple_flowchart()
    fig2.savefig('reports/experiment_1_flowchart_simple.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/experiment_1_flowchart_simple.png")
    
    # LaTeX TikZ code
    print("3. Generating LaTeX TikZ code...")
    tikz = create_latex_tikz()
    with open('reports/experiment_1_flowchart.tex', 'w') as f:
        f.write(tikz)
    print("   ✓ Saved: reports/experiment_1_flowchart.tex")
    
    plt.close('all')
    
    print("\n✅ All flowcharts generated successfully!")
    print("\nGenerated files:")
    print("  - experiment_1_flowchart_detailed.png (detailed version with legend)")
    print("  - experiment_1_flowchart_simple.png (compact version)")
    print("  - experiment_1_flowchart.tex (LaTeX TikZ code)")

if __name__ == "__main__":
    main()

