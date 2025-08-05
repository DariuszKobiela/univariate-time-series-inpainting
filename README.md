# univariate-time-series-inpainting

Univariate time series inpainting



\# Quick test (5-10 minutes)

python run\_improved\_experiment.py --quick



\# Medium experiment (30-45 minutes) 

python run\_improved\_experiment.py --medium



\# Full experiment (30 hours)

python run\_improved\_experiment.py --full



\# Custom configuration

python run\_improved\_experiment.py --iterations 5 --inpainting\_models gaf-unet mtf-unet --forecasting\_models XGBoost Prophet



python run\_improved\_experiment.py --iterations 25 --inpainting\_models gaf-unet mtf-unet rp-unet spec-unet --forecasting\_models XGBoost



python run\_improved\_experiment.py --iterations 15 --inpainting\_models gaf-unet --forecasting\_models XGBoost



