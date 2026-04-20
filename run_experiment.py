import os
import json
import torch
import warnings
import numpy as np
import argparse
warnings.filterwarnings('ignore')

from src.utils import load_config, set_seed
from src.data_access import get_session
from src.preprocessing import extract_units, build_population_activity
from src.features import build_stimulus_signal
from src.datasets import create_dataloaders
from src.models import PersistenceBaseline, LinearBaseline, MLPBaseline, MechanisticModel, HybridModel, LatentCTRNN, LSTMBaseline
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_predictions, plot_loss_curves, plot_phase, plot_error, plot_residual_effect, plot_latent_trajectories

def main():
    parser = argparse.ArgumentParser(description="Run Neural Dynamics Experiment")
    parser.add_argument('--mode', type=str, choices=['lightweight', 'heavyweight'], default='heavyweight',
                        help='Choose lightweight for quick debugging or heavyweight for full training.')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to run (e.g., HybridModel). If not specified, runs all models.')
    args = parser.parse_args()

    config = load_config('config/default.yaml')
    
    if args.mode == 'lightweight':
        print("Running in LIGHTWEIGHT mode. Overriding epochs to 5 and seq_len to 15.")
        config['training']['epochs'] = 5
        config['training']['sequence_length'] = 15
        if args.model == 'HybridModel':
            print("HybridModel lightweight override: using 2 epochs, seq_len 15, batch_size 4096 for a fast stabilization pass.")
            config['training']['epochs'] = 2
            config['training']['sequence_length'] = 15
            config['training']['batch_size'] = 4096
            config['training']['early_stopping_patience'] = 2
        
    set_seed(config['project']['seed'])
    
    # Create output directories
    out_dir = config['project']['output_dir']
    for sub in ['figures', 'metrics', 'models', 'logs']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    session = get_session(config['data']['session_id'], config['project']['data_dir'])
    
    # Time bounds (using full session or a slice)
    stim_table = session.stimulus_presentations
    t_start = stim_table['start_time'].min()
    t_stop = stim_table['stop_time'].max()
    print(f"Global time window: {t_start:.2f} to {t_stop:.2f} seconds")
    
    # 2. Extract units
    exc_ids, inh_ids = extract_units(
        session, 
        region_acronym=config['data']['region_acronym'],
        min_units=config['data']['min_units']
    )
    
    # 3. Build Population Activity x(t) [E_t, I_t]
    time_vector, x_t = build_population_activity(
        session, exc_ids, inh_ids, t_start, t_stop,
        bin_size=config['data']['bin_size'],
        smoothing_sigma=config['data']['smoothing_sigma']
    )
    
    # 4. Build Input Signal u(t)
    _, u_t = build_stimulus_signal(
        session, t_start, t_stop,
        bin_size=config['data']['bin_size'],
        stimulus_class=config['features']['stimulus_class']
    )
    
    # Normalize x_t for easier modeling
    x_mean = x_t.mean(axis=0)
    x_std = x_t.std(axis=0) + 1e-8
    x_t = (x_t - x_mean) / x_std
    
    print(f"Built signals: {len(time_vector)} bins. x(t) shape={x_t.shape}")
    
    # 5. Datasets
    seq_len = config['training'].get('sequence_length', 50)
    train_loader, val_loader, test_loader, test_dataset = create_dataloaders(
        x_t, u_t, time_vector,
        train_split=config['training']['train_split'],
        val_split=config['training']['val_split'],
        batch_size=config['training']['batch_size'],
        sequence_length=seq_len
    )
    
    # We define our models to run dynamically
    models_to_run = {
        'Persistence': PersistenceBaseline(),
        'Linear_Baseline': LinearBaseline(input_dim=3)
    }
    
    if 'baseline_mlp' in config['models']:
        models_to_run['MLP_Baseline'] = MLPBaseline(
            input_dim=3,
            hidden_dims=config['models']['baseline_mlp']['hidden_dims']
        )
        
    m_config = config['models'].get('mechanistic', {})
    if config['models'].get('use_mechanistic', True):
        models_to_run['LatentCTRNN'] = LatentCTRNN(
            hidden_dim=64, dt=m_config.get('dt', 0.01)
        )
        
    if config['models'].get('use_residual', True):
        models_to_run['LSTMBaseline'] = LSTMBaseline(
            hidden_dim=64
        )
        models_to_run['HybridModel'] = HybridModel(
            dt=m_config.get('dt', 0.01),
            activation_function=m_config.get('activation_function', 'sigmoid'),
            residual_hidden_dims=config['models'].get('hybrid', {}).get('residual_hidden_dims', [16])
        )
    
    results = {}
    
    if args.model:
        if args.model in models_to_run:
            models_to_run = {args.model: models_to_run[args.model]}
            print(f"Restricted execution to target model: {args.model}")
        else:
            print(f"Warning: Model '{args.model}' not found. Valid models are: {list(models_to_run.keys())}")
            return
    stored_preds = {}
    
    for name, model in models_to_run.items():
        save_path = os.path.join(out_dir, 'models', f"{name}.pth")
                    
        print(f"\n--- Training {name} ---")
        
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, config, save_path, device
        )
        
        plot_loss_curves(train_losses, val_losses, name, 
                         save_path=os.path.join(out_dir, 'figures', f"{name}_loss_curves.png"))
        
        eval_limit = None
        if args.mode == 'lightweight' and name == 'HybridModel':
            eval_limit = 5000

        metrics, all_preds, all_targets, all_E, all_I = evaluate_model(
            trained_model,
            test_dataset,
            device,
            max_eval_steps=eval_limit
        )
        results[name] = metrics
        stored_preds[name] = all_preds

        test_time = test_dataset.time[1 : 1 + len(all_preds)]
        test_u_t = test_dataset.u.numpy().flatten()[:len(test_time)]
        
        # 1. Overlay Plot & Stimulus Alignment
        plot_predictions(
            test_time, all_targets, all_preds, test_u_t, name,
            save_path=os.path.join(out_dir, 'figures', f"{name}_predictions.png"),
            window_size=500
        )
        
        # 2. Error over time
        plot_error(
            test_time, all_targets, all_preds, name,
            save_path=os.path.join(out_dir, 'figures', f"{name}_error.png"),
            window_size=500
        )
        
        # 3. Phase Plot & Latent Trajectories
        if all_E is not None and all_I is not None:
            plot_phase(
                all_E, all_I, name,
                save_path=os.path.join(out_dir, 'figures', f"{name}_phase.png"),
                window_size=2000
            )
            plot_latent_trajectories(
                test_time, all_E, all_I, name,
                save_path=os.path.join(out_dir, 'figures', f"{name}_latent_trajectories.png"),
                window_size=1000
            )
            
    # 4. Residual Effect Overlay (if both mechanistic and hybrid are present)
    if 'Mechanistic' in stored_preds and 'Hybrid' in stored_preds:
        plot_residual_effect(
            test_time, 
            all_targets, 
            stored_preds['Mechanistic'], 
            stored_preds['Hybrid'],
            save_path=os.path.join(out_dir, 'figures', "residual_effect_overlay.png"),
            window_size=500
        )
        
    print("\n--- Final Results ---")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
        print(f"  Pearson={metrics['Pearson']:.4f}, Cross_Corr={metrics['Cross_Corr']:.4f}, Lag={metrics['Lag']}")
        print(f"  dx/dt Corr={metrics['Deriv_Consistency']:.4f}")
        
    with open(os.path.join(out_dir, 'metrics', 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
