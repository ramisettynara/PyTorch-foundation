import sys

try:
    import torch
    print("✅ PyTorch installed successfully!")
    print(f"   • PyTorch Version: {torch.__version__}")
    print(f"   • Python Version:  {sys.version.split()[0]}")
    
    # Check for Apple Silicon GPU acceleration (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("   • Hardware Accelerator: Apple Silicon GPU (MPS) is AVAILABLE")
        device = torch.device("mps")
    # Check for Nvidia GPU acceleration (CUDA)
    elif torch.cuda.is_available():
        print("   • Hardware Accelerator: Nvidia GPU (CUDA) is AVAILABLE")
        device = torch.device("cuda")
    # Fallback to CPU
    else:
        print("   • Hardware Accelerator: None (Running on CPU)")
        device = torch.device("cpu")

    # Run a quick mathematical test matrix to confirm it works
    x = torch.rand(3, 3, device=device)
    print(f"\n🚀 Basic Tensor Test Successful!\nCreated a random 3x3 matrix on {device.type.upper()}:\n{x}")

except ImportError:
    print("❌ PyTorch is NOT installed in this Python environment.")
    print("Please check your active virtual environment or run: pip install torch")
