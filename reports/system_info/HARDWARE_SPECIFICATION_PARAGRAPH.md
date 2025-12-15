# Hardware and Software Specification - Paragraph for Scientific Article

## Version 1: Concise (for methods section)

All experiments were conducted on a workstation equipped with an AMD Ryzen Threadripper 2920X 12-Core processor (24 logical cores, 3.5 GHz max frequency), 62.75 GB RAM, and an NVIDIA TITAN RTX GPU (24 GB GDDR6, driver version 525.147.05) running Ubuntu 18.04.1 LTS (Linux kernel 5.4.0-150-generic). The experimental framework was implemented in Python 3.10.18, utilizing key libraries including NumPy, pandas, scikit-learn, PyTorch, TensorFlow, XGBoost, statsmodels, and Darts for time series analysis. Code development was facilitated through Cursor IDE with AI-assisted programming capabilities using large language models.

---

## Version 2: Standard (balanced detail)

All computational experiments were performed on a dedicated workstation featuring an AMD Ryzen Threadripper 2920X processor with 12 physical cores (24 threads) operating at up to 3.5 GHz, 62.75 GB of system memory, and an NVIDIA TITAN RTX graphics processing unit with 24 GB of GDDR6 memory (NVIDIA driver 525.147.05, CUDA support). The system operated on Ubuntu 18.04.1 LTS with Linux kernel 5.4.0-150-generic. The entire computational pipeline was implemented in Python 3.10.18, leveraging established scientific computing libraries: NumPy and pandas for data manipulation, scikit-learn for classical machine learning methods, PyTorch and TensorFlow for deep learning implementations, XGBoost for gradient boosting models, statsmodels and Darts for time series forecasting, and pyts for time series imaging transformations. Additional libraries included scipy for statistical analysis, matplotlib and seaborn for visualization, and psutil for system monitoring. The experimental codebase was developed using Cursor IDE, an AI-enhanced integrated development environment that employs large language models to assist in code generation and debugging.

---

## Version 3: Detailed (comprehensive for reproducibility)

### 3.1 Hardware Configuration

All experiments were conducted on a high-performance workstation with the following specifications:

**Central Processing Unit:** AMD Ryzen Threadripper 2920X featuring 12 physical cores with simultaneous multithreading (24 logical processors), base clock frequency of 3.0 GHz with maximum boost frequency of 3.5 GHz, 32 MB L3 cache, and x86_64 architecture.

**Graphics Processing Unit:** NVIDIA TITAN RTX based on Turing architecture (TU102), equipped with 4,608 CUDA cores, 576 second-generation Tensor Cores, 72 RT Cores, 24 GB GDDR6 memory with 384-bit memory interface (672 GB/s bandwidth), and TDP of 280W. The GPU was managed through NVIDIA driver version 525.147.05 with CUDA 12.0 support.

**Memory and Storage:** 62.75 GB DDR4 system memory and 915.82 GB NVMe storage.

**Operating System:** Ubuntu 18.04.1 LTS (Bionic Beaver) with Linux kernel version 5.4.0-150-generic, 64-bit architecture.

### 3.2 Software Environment

The experimental framework was implemented entirely in Python 3.10.18, executed within an Anaconda virtual environment (conda environment: timeseries) to ensure dependency isolation and reproducibility. The following libraries constituted the core computational stack:

**Data Processing and Numerical Computing:** NumPy (array operations and numerical computations), pandas (data frame manipulation and time series handling), scipy (statistical functions and signal processing).

**Machine Learning and Statistical Modeling:** scikit-learn (classical machine learning algorithms including k-nearest neighbors, support vector machines), XGBoost (gradient boosting implementation), statsmodels (ARIMA, SARIMAX time series models), prophet (Facebook's time series forecasting).

**Deep Learning Frameworks:** PyTorch (neural network implementations, GPU acceleration), TensorFlow (alternative deep learning framework), pytorch-lightning (high-level PyTorch interface).

**Time Series Specific Tools:** Darts (univariate and multivariate time series forecasting), pyts (time series to image transformations: Gramian Angular Fields, Markov Transition Fields, Recurrence Plots), dtaidistance (dynamic time warping).

**Visualization and System Monitoring:** matplotlib and seaborn (scientific plotting), psutil (system resource monitoring).

### 3.3 Development Environment

The experimental codebase was developed using Cursor IDE, a modern integrated development environment that integrates large language models (LLMs) for AI-assisted code generation, refactoring, and debugging. This approach facilitated rapid prototyping and implementation of complex experimental pipelines while maintaining code quality and documentation standards.

---

## Version 4: Compact Table Format

**Table 1: Computational Infrastructure**

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen Threadripper 2920X (12C/24T, 3.5 GHz) |
| **GPU** | NVIDIA TITAN RTX (24 GB GDDR6, 4608 CUDA cores) |
| **RAM** | 62.75 GB DDR4 |
| **OS** | Ubuntu 18.04.1 LTS (kernel 5.4.0-150) |
| **Python** | 3.10.18 (Anaconda environment) |
| **GPU Driver** | NVIDIA 525.147.05 (CUDA 12.0) |
| **Key Libraries** | NumPy, pandas, PyTorch, TensorFlow, XGBoost, scikit-learn, statsmodels, Darts, pyts |
| **Development** | Cursor IDE with LLM-assisted programming |

---

## LaTeX Versions

### LaTeX Version 1 (Inline)

```latex
\subsection{Computational Environment}

All experiments were conducted on a workstation equipped with an AMD Ryzen Threadripper 2920X 12-Core processor (24 logical cores, 3.5~GHz max frequency), 62.75~GB RAM, and an NVIDIA TITAN RTX GPU (24~GB GDDR6, driver version 525.147.05) running Ubuntu 18.04.1 LTS (Linux kernel 5.4.0-150-generic). The experimental framework was implemented in Python 3.10.18, utilizing key libraries including NumPy, pandas, scikit-learn, PyTorch, TensorFlow, XGBoost, statsmodels, and Darts for time series analysis. Code development was facilitated through Cursor IDE with AI-assisted programming capabilities using large language models.
```

### LaTeX Version 2 (Table)

```latex
\begin{table}[htbp]
\centering
\caption{Computational Infrastructure and Software Stack}
\label{tab:hardware}
\begin{tabular}{ll}
\toprule
\textbf{Component} & \textbf{Specification} \\
\midrule
CPU & AMD Ryzen Threadripper 2920X (12C/24T, 3.5 GHz) \\
GPU & NVIDIA TITAN RTX (24 GB GDDR6, 4608 CUDA cores) \\
RAM & 62.75 GB DDR4 \\
Operating System & Ubuntu 18.04.1 LTS (kernel 5.4.0-150) \\
Python Version & 3.10.18 (Anaconda environment) \\
GPU Driver & NVIDIA 525.147.05 (CUDA 12.0) \\
\midrule
\multicolumn{2}{l}{\textbf{Key Python Libraries:}} \\
Data Processing & NumPy, pandas, scipy \\
Machine Learning & scikit-learn, XGBoost, statsmodels \\
Deep Learning & PyTorch, TensorFlow, pytorch-lightning \\
Time Series & Darts, pyts, dtaidistance, prophet \\
Development & Cursor IDE with LLM assistance \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Recommendations

- **For conference papers (page limit):** Use **Version 1** or **Version 4** (table)
- **For journal papers:** Use **Version 2**
- **For highly technical venues or reproducibility-focused journals:** Use **Version 3**
- **For submission with strict formatting:** Use **LaTeX versions**

---

## Additional Notes

### GPU Specifications (for supplementary materials):
- Architecture: Turing (TU102)
- Compute Capability: 7.5
- Tensor Cores: 576 (2nd gen)
- RT Cores: 72
- FP32 Performance: ~16.3 TFLOPS
- FP16 Performance: ~130.5 TFLOPS (with Tensor Cores)
- Memory Bandwidth: 672 GB/s
- TDP: 280W

### System Performance During Experiments:
- CPU utilization: 0.4% (idle) to variable under load
- Memory utilization: 12.6% baseline
- GPU utilization: 0% idle, up to 100% during training
- Experiment start date: 2025-11-29
- System uptime from: 2025-07-16

### Reproducibility Statement (optional addition):
All experimental code, configuration files, and trained models are available in the project repository. The computational environment can be reproduced using the provided \texttt{requirements.txt} file and Anaconda environment specification.

