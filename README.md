# Univariate Time Series Inpainting

Comparison of deep learning image inpainting methods vs traditional imputation for time series forecasting.

## Quick Start

### Quick test (5-10 minutes)
```bash
python run_improved_experiment.py --quick
```

### Medium experiment (30-45 minutes)
```bash
python run_improved_experiment.py --medium
```

### Full experiment (30 hours)
```bash
python run_improved_experiment.py --full
```

### Custom configuration
```bash
# Example: 5 iterations with selected models
python run_improved_experiment.py --iterations 5 --inpainting_models gaf-unet mtf-unet --forecasting_models XGBoost Prophet

# Example: 25 iterations, all inpainting models, XGBoost only
python run_improved_experiment.py --iterations 25 --inpainting_models gaf-unet mtf-unet rp-unet spec-unet --forecasting_models XGBoost

# Example: 15 iterations, single model comparison
python run_improved_experiment.py --iterations 15 --inpainting_models gaf-unet --forecasting_models XGBoost
```

## Running Long Experiments with tmux

For long-running experiments, use `tmux` to keep your session alive even if you disconnect from SSH or close your terminal.

### Initial Setup (first time only)
```bash
# Install tmux if not already installed
sudo apt-get update
sudo apt-get install tmux
```

### Starting an Experiment in tmux

**1. Create a new tmux session:**
```bash
tmux new -s experiment
```

**2. Run your experiment inside the tmux session:**
```bash
cd /home/darek/univariate-time-series-inpainting
python run_improved_experiment.py --quick
# or --medium, --full, etc.
```

**3. Detach from tmux (leave it running in background):**
- Press `Ctrl+b`, then press `d`
- Your experiment continues running!

### Managing tmux Sessions

**Reconnect to your running experiment:**
```bash
tmux attach -t experiment
```

**List all tmux sessions:**
```bash
tmux ls
```

**Create multiple experiments (different sessions):**
```bash
tmux new -s experiment1
# Run first experiment...
# Ctrl+b, d to detach

tmux new -s experiment2
# Run second experiment...
# Ctrl+b, d to detach
```

**Kill a session when done:**
```bash
tmux kill-session -t experiment
```

### Useful tmux Commands

Inside tmux session:
- `Ctrl+b` then `d` - Detach (keep running in background)
- `Ctrl+b` then `c` - Create new window
- `Ctrl+b` then `n` - Next window
- `Ctrl+b` then `p` - Previous window
- `Ctrl+b` then `,` - Rename window
- `exit` - Exit current window/session

### Complete Workflow Example

```bash
# 1. Start tmux session
tmux new -s my_experiment

# 2. Navigate to project
cd /home/darek/univariate-time-series-inpainting

# 3. Start experiment
python run_improved_experiment.py --full

# 4. Watch it start, then detach
# Press: Ctrl+b, then d

# 5. Close your SSH connection - experiment keeps running!

# --- Later (hours/days later) ---

# 6. Reconnect to check progress
tmux attach -t my_experiment

# 7. Monitor results while it runs
watch -n 10 'wc -l results/quick_experiment/df_final.csv'

# 8. When done, exit tmux
exit
```

### Monitoring Progress

While experiment runs (from outside tmux):
```bash
# Check how many results have been saved
wc -l results/quick_experiment/df_final.csv

# Watch results grow in real-time
watch -n 10 'wc -l results/quick_experiment/df_final.csv'

# Check if experiment is still running
ps aux | grep python

# View recent output (if logging to file)
tail -f experiment.log
```

### Pro Tips

1. **Name your sessions meaningfully:**
   ```bash
   tmux new -s xgboost_full
   tmux new -s prophet_test
   ```

2. **Always detach before closing terminal:**
   - Use `Ctrl+b` then `d` instead of closing terminal
   
3. **Save output to log file:**
   ```bash
   python run_improved_experiment.py --full 2>&1 | tee experiment.log
   ```

4. **Check if tmux session exists before creating:**
   ```bash
   tmux has-session -t experiment 2>/dev/null && echo "Session exists" || tmux new -s experiment
   ```

5. **Results are saved incrementally:**
   - Check `results/quick_experiment/df_final.csv` anytime
   - Results persist even if experiment crashes



