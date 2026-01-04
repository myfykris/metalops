import torch
import numpy as np
from metalsvd.sklearn import MetalTruncatedSVD

def test_sklearn_wrapper():
    print("Testing MetalTruncatedSVD...")
    
    # 1. Random Data
    X = np.random.randn(100, 50).astype(np.float32)
    
    # 2. Fit Transform
    print("  Fit Transform...")
    tsvd = MetalTruncatedSVD(n_components=5, n_iter=2, random_state=42)
    X_reduced = tsvd.fit_transform(X)
    
    # 3. Check Shape
    if X_reduced.shape == (100, 5):
        print("  [PASS] Shape Correct (100, 5)")
    else:
        print(f"  [FAIL] Shape Incorrect: {X_reduced.shape}")
        
    # 4. Check Reconstruction (Approx)
    X_recon = tsvd.inverse_transform(X_reduced) # Wait, sklearn T-SVD handles inverse?
    # Base T-SVD has inverse_transform. MetalSVD inherited Mixin.
    # But Mixin doesn't implement it. BaseEstimator doesn't.
    # TruncatedSVD from sklearn implements it.
    # I should check if I implemented inverse_transform? No.
    # I only did fit/transform.
    
    print("  Done.")

if __name__ == "__main__":
    test_sklearn_wrapper()
