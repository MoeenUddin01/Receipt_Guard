import sys
from pathlib import Path
from typing import Union, Any, Dict

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))

try:
    from src.config import CFG, override_config
    from src.pipelines.model_training_pipeline import run_training_pipeline, TrainingConfig, normalize_config
    from src.pipelines.preprocessing_pipeline import run_preprocessing_pipeline, PreprocessingConfig
    from src.pipelines.evaluation_pipeline import run_evaluation_pipeline, EvaluationConfig
    
    print("✅ All pipelines imported successfully!")
    
    # Test normalization logic
    print("\nTesting configuration normalization...")
    
    # 1. TrainingConfig
    t_config = normalize_config(CFG)
    print(f"  TrainingConfig data_path: {t_config.data_path}")
    print(f"  TrainingConfig model_path: {t_config.model_path}")
    
    # 2. PreprocessingConfig
    p_config = PreprocessingConfig() # Already has normalization in __post_init__
    print(f"  PreprocessingConfig raw_data_path: {p_config.raw_data_path}")
    
    # 3. EvaluationConfig
    # Use a dummy checkpoint path
    e_config = EvaluationConfig(checkpoint_path="dummy.pt")
    print(f"  EvaluationConfig model_path: {e_config.model_path}")
    
    print("\n✅ Configuration normalization looks good!")
    
    print("\nTesting attribute access in normalized TrainingConfig...")
    # This was the cause of the previous error
    try:
        model_path = t_config.model_path
        print(f"  Successfully accessed t_config.model_path: {model_path}")
    except AttributeError as e:
        print(f"  ❌ FAILED: {e}")
        sys.exit(1)

    print("\n✅ Verification complete! No structural errors found.")

except Exception as e:
    print(f"\n❌ ERROR during verification: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
