from typing import Dict, Any

class ExperimentConfig:
    """Defines hyperparameter search spaces for Optuna and experiment metadata."""

    OPTUNA_STUDY_NAME: str = "ppo_trading_optimization"
    OPTUNA_N_TRIALS: int = 30
    OPTUNA_TIMEOUT: int = 3600  # seconds

    # Search spaces (low, high, log)
    SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
        "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
        "n_steps": {"low": 512, "high": 4096, "step": 512, "log": False},
        "batch_size": {"low": 32, "high": 256, "step": 32, "log": False},
        "n_epochs": {"low": 5, "high": 20, "step": 1, "log": False},
        "gamma": {"low": 0.9, "high": 0.999, "log": False},
        "gae_lambda": {"low": 0.9, "high": 0.99, "log": False},
        "clip_range": {"low": 0.1, "high": 0.4, "log": False},
        "ent_coef": {"low": 1e-4, "high": 0.1, "log": True},
        "vf_coef": {"low": 0.2, "high": 0.8, "log": False},
        "max_grad_norm": {"low": 0.3, "high": 1.0, "log": False},
    }

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Return a dictionary of default hyperparameters (from Config)."""
        from config.config import Config
        return {
            "learning_rate": Config.PPO_LEARNING_RATE,
            "n_steps": Config.PPO_N_STEPS,
            "batch_size": Config.PPO_BATCH_SIZE,
            "n_epochs": Config.PPO_N_EPOCHS,
            "gamma": Config.PPO_GAMMA,
            "gae_lambda": Config.PPO_GAE_LAMBDA,
            "clip_range": Config.PPO_CLIP_RANGE,
            "ent_coef": Config.PPO_ENT_COEF,
            "vf_coef": Config.PPO_VF_COEF,
            "max_grad_norm": Config.PPO_MAX_GRAD_NORM,
        }