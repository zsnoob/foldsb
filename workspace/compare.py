import torch
import pickle
import numpy as np
import os

def try_load_tensor(filepath):
    """尝试多种方法加载tensor文件"""
    print(f"Trying to load: {filepath}")
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"❌ File {filepath} does not exist")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(filepath)
    print(f"📁 File size: {file_size} bytes")
    
    # 方法1: 标准torch.load
    try:
        tensor = torch.load(filepath, map_location='cpu')
        print(f"✅ torch.load successful - type: {type(tensor)}")
        if hasattr(tensor, 'shape'):
            print(f"   Shape: {tensor.shape}, dtype: {tensor.dtype}")
        return tensor
    except Exception as e:
        print(f"❌ torch.load failed: {e}")
    
    # 方法2: 尝试不同的权重加载方式
    try:
        tensor = torch.load(filepath, map_location='cpu', weights_only=True)
        print(f"✅ torch.load (weights_only=True) successful")
        return tensor
    except Exception as e:
        print(f"❌ torch.load (weights_only=True) failed: {e}")
    
    # 方法3: 尝试pickle加载
    try:
        with open(filepath, 'rb') as f:
            tensor = pickle.load(f)
        print(f"✅ pickle.load successful - type: {type(tensor)}")
        return tensor
    except Exception as e:
        print(f"❌ pickle.load failed: {e}")
    
    # 方法4: 尝试numpy加载(如果是numpy保存的)
    try:
        array = np.load(filepath, allow_pickle=True)
        tensor = torch.from_numpy(array)
        print(f"✅ numpy.load successful")
        return tensor
    except Exception as e:
        print(f"❌ numpy.load failed: {e}")
    
    # 方法5: 检查文件头
    try:
        with open(filepath, 'rb') as f:
            header = f.read(100)
        print(f"📄 File header (first 100 bytes):")
        print(f"   Hex: {header.hex()[:50]}...")
        print(f"   ASCII: {header[:50]}")
    except Exception as e:
        print(f"❌ Cannot read file header: {e}")
    
    return None

def compare_tensors_safe(tensor1, tensor2, name1="nopipe", name2="pipe"):
    """安全的tensor比较，处理各种边界情况"""
    
    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")
    
    # 检查是否成功加载
    if tensor1 is None:
        print(f"❌ {name1} failed to load")
        return None
    if tensor2 is None:
        print(f"❌ {name2} failed to load")
        return None
    
    # 转换为tensor（如果需要）
    if not isinstance(tensor1, torch.Tensor):
        print(f"Converting {name1} to tensor...")
        tensor1 = torch.tensor(tensor1) if hasattr(tensor1, '__iter__') else tensor1
    
    if not isinstance(tensor2, torch.Tensor):
        print(f"Converting {name2} to tensor...")
        tensor2 = torch.tensor(tensor2) if hasattr(tensor2, '__iter__') else tensor2
    
    # 基本信息
    print(f"\n📊 Basic Info:")
    print(f"{name1:10} - type: {type(tensor1)}")
    print(f"{name2:10} - type: {type(tensor2)}")
    
    if hasattr(tensor1, 'shape') and hasattr(tensor2, 'shape'):
        print(f"{name1:10} - shape: {tensor1.shape}, dtype: {tensor1.dtype}")
        print(f"{name2:10} - shape: {tensor2.shape}, dtype: {tensor2.dtype}")
        
        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            print(f"❌ Shape mismatch!")
            print(f"   {name1}: {tensor1.shape}")
            print(f"   {name2}: {tensor2.shape}")
            return {'status': 'shape_mismatch'}
        
        # 数值比较
        try:
            # 确保数据类型一致
            if tensor1.dtype != tensor2.dtype:
                print(f"⚠️ Converting dtypes: {tensor1.dtype} -> {tensor2.dtype}")
                tensor1 = tensor1.to(tensor2.dtype)
            
            # 统计信息
            print(f"\n📈 Statistics:")
            print(f"{name1:10} - mean: {tensor1.mean().item():.6f}, std: {tensor1.std().item():.6f}")
            print(f"{name1:10} - min:  {tensor1.min().item():.6f}, max: {tensor1.max().item():.6f}")
            print(f"{name2:10} - mean: {tensor2.mean().item():.6f}, std: {tensor2.std().item():.6f}")
            print(f"{name2:10} - min:  {tensor2.min().item():.6f}, max: {tensor2.max().item():.6f}")
            
            # 差异分析
            print(f"\n🔍 Difference Analysis:")
            diff = tensor1 - tensor2
            abs_diff = torch.abs(diff)
            
            mean_abs_diff = abs_diff.mean().item()
            max_abs_diff = abs_diff.max().item()
            
            print(f"Mean absolute difference: {mean_abs_diff:.8f}")
            print(f"Max absolute difference:  {max_abs_diff:.8f}")
            
            # 相对误差
            rel_diff = abs_diff / (torch.abs(tensor1) + 1e-8)
            print(f"Mean relative difference: {rel_diff.mean().item():.8f}")
            print(f"Max relative difference:  {rel_diff.max().item():.8f}")
            
            # 相等性检查
            tolerances = [1e-8, 1e-6, 1e-4, 1e-2]
            print(f"\n✅ Equality Check:")
            for tol in tolerances:
                is_close = torch.allclose(tensor1, tensor2, atol=tol, rtol=tol)
                print(f"torch.allclose(atol={tol:g}, rtol={tol:g}): {is_close}")
            
            # 完全相等
            exact_equal = torch.equal(tensor1, tensor2)
            print(f"torch.equal (exact): {exact_equal}")
            
            return {
                'status': 'success',
                'mean_abs_diff': mean_abs_diff,
                'max_abs_diff': max_abs_diff,
                'exact_equal': exact_equal,
                'close_1e6': torch.allclose(tensor1, tensor2, atol=1e-6, rtol=1e-6),
                'close_1e4': torch.allclose(tensor1, tensor2, atol=1e-4, rtol=1e-4)
            }
            
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            return {'status': 'error', 'error': str(e)}
    
    else:
        print("❌ Objects are not tensors or don't have shape attribute")
        return {'status': 'not_tensors'}

def main():
    """主函数"""
    files = ["hidden_states_nopipe_3.pt", "hidden_states_pipe_3.pt"]
    
    # 尝试加载两个文件
    print("🔄 Loading files...")
    tensors = []
    for filepath in files:
        tensor = try_load_tensor(filepath)
        tensors.append(tensor)
    
    # 比较tensor
    if len(tensors) == 2:
        result = compare_tensors_safe(tensors[0], tensors[1], "nopipe", "pipe")
        
        if result and result.get('status') == 'success':
            print(f"\n🎯 Summary:")
            if result['exact_equal']:
                print("✅ Tensors are EXACTLY equal!")
            elif result['close_1e6']:
                print("✅ Tensors are very close (1e-6 tolerance)")
            elif result['close_1e4']:
                print("⚠️ Tensors are somewhat close (1e-4 tolerance)")
            else:
                print("❌ Tensors have significant differences")
                print(f"   Max difference: {result['max_abs_diff']:.8f}")

if __name__ == "__main__":
    main()
