"""
Export script to manually reconstruct the actor newtwork and export it to TorchScript
Example of usage: 
python3 export_policy.py --checkpoint_path model_90000000.pt
"""
import torch
import torch.nn as nn 
import argparse
import os 
import json


class MLP(nn.Module):
    """Network matching the architecture of CMORL sstudent actor"""
    def __init__(self, input_dim, output_dim, hidden_dims, activation="ELU"):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim 

        layers.append(nn.Linear(prev_dim, output_dim))
        self.module_list = nn.ModuleList(layers)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x 


class ActorGaussian(nn.Module):
    """Gaussian actor from CMORL framework"""

    def __init__(self, obs_dim, action_dim, hidden_dims, activation="ELU", log_std_init=0.0):
        super().__init__()
        self.obs_dim = obs_dim 
        self.action_dim = action_dim 

        self.model = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation)

        activation_fn = getattr(nn, activation)
        self.mean_decoder = nn.Sequential(
            activation_fn(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        self.log_std_init = log_std_init

    def forward(self, obs):

        x = self.model(obs)
        mean = self.mean_decoder(x)

        log_std = torch.ones_like(mean) * self.log_std_init 
        std = torch.exp(log_std)

        return mean, log_std, std
         


class PolicyForDeployment(nn.Module):
    def __init__(self, actor, obs_mean, obs_std):
        super().__init__()
        self.actor = actor
        self.register_buffer('obs_mean', obs_mean)
        self.register_buffer('obs_std', obs_std)

    def forward(self, obs):
        # Normalize
        normalized_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        # Get action mean
        mean, _, _ = self.actor(normalized_obs)
        return mean



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to your checkpoint")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for exported policy")
    parser.add_argument("--obs_dim", type=int, default=420, help="Observation dimension")
    parser.add_argument("--action_dim", type=int, default=12, help="Action dimension")
    parser.add_argument("--history_len", type=int, default=10, help="History length")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 128, 64], help="Hidden dimensions")
    args = parser.parse_args() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading a checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    actor = ActorGaussian(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim, 
        hidden_dims=args.hidden_dims,
        activation="ELU",
        log_std_init=0.0,
    ).to(device)

    actor.load_state_dict(checkpoint['actor'])
    actor.eval() 
    print("Loaded actor weights")

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    model_num = os.path.basename(args.checkpoint_path).replace('model_', '').replace('.pt', '')

    obs_mean = torch.zeros(args.obs_dim, device=device)
    obs_std = torch.ones(args.obs_dim, device=device)
    
    deployment_policy = PolicyForDeployment(actor, obs_mean, obs_std) 

    # trace the model
    print(f"Tracing model with input shape: {1, args.obs_dim}")
    dummy_obs = torch.randn(1, args.obs_dim, device=device)
    traced_policy = torch.jit.trace(deployment_policy, dummy_obs)

    # Determine output path
    if args.output_path is None:
        export_dir = os.path.join(os.path.dirname(checkpoint_dir), 'exported')
        parent_dir = os.path.dirname(checkpoint_dir)
        export_dir = os.path.join(parent_dir, 'exported')
        os.makedirs(export_dir, exist_ok=True)
        args.output_path = f"{export_dir}/policy_{model_num}.pt"
        args.output_path = os.path.join(export_dir, f'policy_{model_num}.pt')
        print(f"Using default export directory: {export_dir}")
    else:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:  # Only create directory if path includes a directory
            os.makedirs(output_dir, exist_ok=True)

    traced_policy.save(args.output_path)
    print(f"\n Policy exported successfully to: {args.output_path}")
     
    # Save config
    raw_obs_dim = args.obs_dim // args.history_len
    config = {
        'obs_dim': args.obs_dim,
        'raw_obs_dim': raw_obs_dim,
        'action_dim': args.action_dim,
        'history_len': args.history_len,
        'num_actions': args.action_dim,
        'control_dt': 0.02,
        'default_angles': [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                          0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    }

    config_path = args.output_path.replace('.pt', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to: {config_path}")
    print(f"\nExport info:")
    print(f"  - Input: (batch, {args.obs_dim}) = (batch, {raw_obs_dim} × {args.history_len})")
    print(f"  - Output: (batch, {args.action_dim})")
    print(f"  - Hidden dims: {args.hidden_dims}")

    # Test the exported model
    print(f"\nTesting exported model...")
    test_obs = torch.randn(1, args.obs_dim, device=device)
    with torch.no_grad():
        test_action = traced_policy(test_obs)
    print(f"Test passed! Output shape: {test_action.shape}")
