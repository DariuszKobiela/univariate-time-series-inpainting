# Experiment 1: Description and Methodology

## Enhanced Description for Article

### Version 1: Detailed (for Methods section)

In the first experiment, we evaluate how image-based inpainting models compare to classical time-domain reconstruction methods when recovering missing values in univariate industrial time series. The experimental pipeline consists of four main stages: data preparation, missing data injection, reconstruction, and evaluation.

**Data Preparation.** Three real-world industrial datasets were selected: boiler outlet temperature measurements, pump sensor readings, and vibration sensor data. Each dataset represents a different type of temporal pattern commonly observed in industrial monitoring systems.

**Missing Data Injection.** For each original time series, we systematically introduced missing values following three missingness mechanisms: Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). Three missingness rates were tested: 2%, 5%, and 10%, representing scenarios from minor sensor failures to significant data loss. Each configuration was repeated across 10 independent iterations with different random seeds to ensure statistical robustness, resulting in 270 corrupted series per dataset (3 mechanisms × 3 rates × 10 iterations).

**Reconstruction Methods.** Two categories of methods were evaluated: (1) Classical time-domain approaches including statistical imputation (mean, median, forward/backward fill), interpolation techniques (linear, cubic, spline), k-nearest neighbors, and autoregressive models (SARIMAX); and (2) Image-based inpainting methods utilizing time series imaging transformations—Gramian Angular Field (GAF), Markov Transition Field (MTF), Recurrence Plot (RP), and Spectrogram (SPEC)—combined with deep learning models (U-Net, Stable Diffusion 2). For image-based methods, time series were first transformed into 2D images, inpainted using the deep learning model, and then inverse-transformed back to the time domain.

**Evaluation Protocol.** For each reconstructed series, the locations of missing values were identified using the binary mask applied during corruption. These masked indices were used to extract corresponding segments from the original, corrupted, and reconstructed time series. The absolute differences between original and reconstructed values within these masked regions were computed element-wise and averaged, yielding the Mean Absolute Difference (MAD). Lower MAD values indicate superior reconstruction quality. This evaluation approach ensures that only the model's ability to recover missing values is assessed, not its performance on already-observed data.

---

### Version 2: Concise (for compact papers)

In the first experiment, we evaluate image-based inpainting models against classical time-domain reconstruction methods for recovering missing values in univariate industrial time series. Three real-world datasets (boiler temperature, pump sensor, vibration sensor) were systematically corrupted using three missingness mechanisms (MCAR, MAR, MNAR) at three rates (2%, 5%, 10%), with 10 iterations each (270 configurations per dataset). 

Reconstruction methods included classical approaches (statistical imputation, interpolation, k-NN, SARIMAX) and image-based techniques combining time series imaging transformations (GAF, MTF, RP, Spectrogram) with deep learning models (U-Net, Stable Diffusion 2). For evaluation, binary masks identified missing value locations, enabling extraction of corresponding segments from original and reconstructed series. The Mean Absolute Difference (MAD) between these segments was computed and averaged. Lower MAD indicates better reconstruction quality.

---

### Version 3: Technical (with equations)

**Experimental Design.** Let $\mathcal{D} = \{d_1, d_2, d_3\}$ represent three industrial time series datasets. For each dataset $d_i$, we generate corrupted versions $\tilde{d}_i$ by applying missingness mechanisms $m \in \{\text{MCAR}, \text{MAR}, \text{MNAR}\}$ with rates $r \in \{0.02, 0.05, 0.10\}$. Each configuration is repeated $k=10$ times with different random seeds.

**Reconstruction.** Given a corrupted series $\tilde{x} \in \mathbb{R}^T$ with binary mask $M \in \{0,1\}^T$ (where $M_t=1$ indicates observed values), reconstruction methods produce $\hat{x} \in \mathbb{R}^T$. Classical methods operate directly in the time domain: $\hat{x} = f_{\text{classical}}(\tilde{x}, M)$. Image-based methods follow a three-step pipeline: (1) transformation $I = \phi(\tilde{x})$ where $\phi: \mathbb{R}^T \to \mathbb{R}^{H \times W}$, (2) inpainting $\hat{I} = f_{\text{inpaint}}(I, \phi(M))$, and (3) inverse transformation $\hat{x} = \phi^{-1}(\hat{I})$.

**Evaluation Metric.** Let $x \in \mathbb{R}^T$ denote the original series and $\hat{x}$ the reconstruction. The Mean Absolute Difference (MAD) is computed only over missing locations:

$$\text{MAD} = \frac{1}{|\{t : M_t = 0\}|} \sum_{t : M_t = 0} |x_t - \hat{x}_t|$$

where $M_t=0$ indicates originally missing values. Lower MAD indicates superior reconstruction quality.

---

## LaTeX Versions

### LaTeX Version 1: Detailed

```latex
\subsection{Experiment 1: Time Series Reconstruction Evaluation}

In the first experiment, we evaluate how image-based inpainting models compare to classical time-domain reconstruction methods when recovering missing values in univariate industrial time series. The experimental pipeline consists of four main stages: data preparation, missing data injection, reconstruction, and evaluation.

\textbf{Data Preparation.} Three real-world industrial datasets were selected: boiler outlet temperature measurements, pump sensor readings, and vibration sensor data. Each dataset represents a different type of temporal pattern commonly observed in industrial monitoring systems.

\textbf{Missing Data Injection.} For each original time series, we systematically introduced missing values following three missingness mechanisms: Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). Three missingness rates were tested: 2\%, 5\%, and 10\%, representing scenarios from minor sensor failures to significant data loss. Each configuration was repeated across 10 independent iterations with different random seeds to ensure statistical robustness, resulting in 270 corrupted series per dataset ($3 \times 3 \times 10$).

\textbf{Reconstruction Methods.} Two categories of methods were evaluated: (1)~Classical time-domain approaches including statistical imputation (mean, median, forward/backward fill), interpolation techniques (linear, cubic, spline), k-nearest neighbors, and autoregressive models (SARIMAX); and (2)~Image-based inpainting methods utilizing time series imaging transformations---Gramian Angular Field (GAF), Markov Transition Field (MTF), Recurrence Plot (RP), and Spectrogram (SPEC)---combined with deep learning models (U-Net, fine-tuned Stable Diffusion~2). For image-based methods, time series were first transformed into 2D images, inpainted using the deep learning model, and then inverse-transformed back to the time domain.

\textbf{Evaluation Protocol.} For each reconstructed series, the locations of missing values were identified using the binary mask applied during corruption. These masked indices were used to extract corresponding segments from the original and reconstructed time series. The absolute differences between original and reconstructed values within these masked regions were computed element-wise and averaged, yielding the Mean Absolute Difference (MAD), reported in Tables~\ref{tab:mad_advanced} and~\ref{tab:mad_simple}. Lower MAD values indicate superior reconstruction quality. This evaluation approach ensures that only the model's ability to recover missing values is assessed, not its performance on already-observed data.
```

### LaTeX Version 2: With Equations

```latex
\subsection{Experiment 1: Evaluation Methodology}

\textbf{Experimental Design.} Let $\mathcal{D} = \{d_1, d_2, d_3\}$ represent three industrial time series datasets. For each dataset $d_i$, we generate corrupted versions $\tilde{d}_i$ by applying missingness mechanisms $m \in \{\text{MCAR}, \text{MAR}, \text{MNAR}\}$ with rates $r \in \{0.02, 0.05, 0.10\}$. Each configuration is repeated $k=10$ times with different random seeds, yielding 270 configurations per dataset.

\textbf{Reconstruction.} Given a corrupted series $\tilde{x} \in \mathbb{R}^T$ with binary mask $M \in \{0,1\}^T$ (where $M_t=1$ indicates observed values), reconstruction methods produce $\hat{x} \in \mathbb{R}^T$. Classical methods operate directly in the time domain: $\hat{x} = f_{\text{classical}}(\tilde{x}, M)$. Image-based methods follow a three-step pipeline:
\begin{enumerate}[leftmargin=*]
    \item Transformation: $I = \phi(\tilde{x})$ where $\phi: \mathbb{R}^T \to \mathbb{R}^{H \times W}$
    \item Inpainting: $\hat{I} = f_{\text{inpaint}}(I, \phi(M))$
    \item Inverse transformation: $\hat{x} = \phi^{-1}(\hat{I})$
\end{enumerate}

\textbf{Evaluation Metric.} Let $x \in \mathbb{R}^T$ denote the original series and $\hat{x}$ the reconstruction. The Mean Absolute Difference (MAD) is computed only over missing locations:
\begin{equation}
\text{MAD} = \frac{1}{|\{t : M_t = 0\}|} \sum_{t : M_t = 0} |x_t - \hat{x}_t|
\end{equation}
where $M_t=0$ indicates originally missing values. Lower MAD indicates superior reconstruction quality, as reported in Tables~\ref{tab:mad_advanced} and~\ref{tab:mad_simple}.
```

---

## Key Improvements Over Original:

1. **Structured into clear stages**: Data Preparation → Injection → Reconstruction → Evaluation
2. **Added dataset details**: Specific types of industrial data
3. **Quantified experimental design**: 270 configurations per dataset (3×3×10)
4. **Explained missingness mechanisms**: MCAR, MAR, MNAR with rationale
5. **Detailed reconstruction methods**: Clear separation of classical vs. image-based
6. **Pipeline explanation**: Three-step process for image-based methods
7. **Clarified evaluation**: Only missing values assessed, not observed data
8. **Statistical robustness**: Mentioned random seeds and iterations
9. **Added mathematical formulation**: For technical audiences
10. **Professional terminology**: "corruption", "segments", "element-wise"

