"""
MSim X-ray Simulation with PV Streaming

This module properly streams projections to PVAccess viewers as they are computed.
"""

import numpy as np
import cupy as cp
from msim.physics import projection
from msim.LSim_wrap import rotate_volume, build_quaternion
import pvaccess as pva
import time


class StreamingManager:
    """Manages PV streaming for simulations."""
    
    def __init__(self):
        self.servers = {}
        self.counters = {}
    
    def setup_pv(self, pv_name, initial_image=None):
        """Setup a PV for streaming."""
        if pv_name in self.servers:
            return  # Already setup
        
        # Create initial dummy image if not provided
        if initial_image is None:
            initial_image = np.zeros((256, 256), dtype=np.uint8)
        
        # Import the utility
        from msim.stream import AdImageUtility
        
        # Create NTNDArray structure
        nt = AdImageUtility.generateNtNdArray2D(0, initial_image)
        
        # Create PVA server
        server = pva.PvaServer()
        server.addRecord(pv_name, nt)
        
        self.servers[pv_name] = {
            'server': server,
            'nt': nt,
            'utility': AdImageUtility
        }
        self.counters[pv_name] = 1
        
        print(f"✓ Created PV: {pv_name}")
    
    def update_pv(self, pv_name, image):
        """Update a PV with new image data."""
        if pv_name not in self.servers:
            # Auto-create if it doesn't exist
            self.setup_pv(pv_name, image)
            return
        
        server_info = self.servers[pv_name]
        nt = server_info['nt']
        utility = server_info['utility']
        uid = self.counters[pv_name]
        
        # Ensure uint8 2D array
        if image.dtype != np.uint8:
            # Normalize to 0-255
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        
        # Update the NTNDArray
        utility.replaceNtNdArrayImage2D(nt, uid, image)
        server_info['server'].update(pv_name, nt)
        
        # Increment counter
        self.counters[pv_name] += 1
    
    def cleanup(self):
        """Cleanup all servers."""
        self.servers.clear()
        self.counters.clear()


# Global streaming manager
_STREAM_MANAGER = StreamingManager()


class GPUVolumeManager:
    """Manages GPU memory for volume rotations efficiently."""
    
    def __init__(self, volume_labels, lookup, voxel_size, config):
        self.volume_shape = volume_labels.shape
        self.lookup = lookup
        self.voxel_size = voxel_size
        self.config = config
        
        # Keep original volume on GPU
        self.volume_gpu = cp.asarray(volume_labels, dtype=cp.float32)
        self.rotated_gpu = cp.empty_like(self.volume_gpu)
        
        print(f"GPU Volume Manager initialized: {self.volume_shape}")
        print(f"GPU memory used: {self.volume_gpu.nbytes / 1e9:.2f} GB")
    
    def rotate_and_project(self, rotation_deg, tilt_deg=0):
        """Rotate volume on GPU and simulate projection."""
        # Build quaternion
        tilt_rad = np.deg2rad(tilt_deg)
        theta_rad = np.deg2rad(rotation_deg)
        quat = build_quaternion(tilt_rad, theta_rad)
        
        # Rotate volume
        volume_cpu = cp.asnumpy(self.volume_gpu)
        rotated_cpu = np.empty_like(volume_cpu)
        volume_contiguous = np.ascontiguousarray(volume_cpu, dtype=np.float32)
        rotated_contiguous = np.ascontiguousarray(rotated_cpu, dtype=np.float32)
        
        rotate_volume(volume_contiguous, rotated_contiguous, quat)
        rotated_int = rotated_contiguous.astype(np.int32)
        
        # Project (this happens on GPU in physics.projection)
        return projection(rotated_int, self.lookup, self.voxel_size, self.config)
    
    def cleanup(self):
        """Free GPU memory."""
        del self.volume_gpu
        del self.rotated_gpu
        cp.get_default_memory_pool().free_all_blocks()


def simulate_tomography_projection(volume_labels, lookup, voxel_size, rotation_deg, 
                                   config, stream_pv=None):
    """
    Simulate tomography projection - optionally stream to PV.
    
    Parameters
    ----------
    volume_labels : np.ndarray
        Volume to project
    lookup : dict
        Material lookup table
    voxel_size : float
        Voxel size
    rotation_deg : float
        Rotation angle in degrees
    config : object
        Simulation configuration
    stream_pv : str, optional
        PV name to stream to (e.g., 'SIMTP:IMG'). If None, no streaming.
    
    Returns
    -------
    np.ndarray
        Projection image
    """
    # Build quaternion
    quat = build_quaternion(0.0, np.deg2rad(rotation_deg))
    
    # Rotate volume
    rotated = np.empty_like(volume_labels, dtype=volume_labels.dtype)
    volume_contiguous = np.ascontiguousarray(volume_labels, dtype=np.float32)
    rotated_contiguous = np.ascontiguousarray(rotated, dtype=np.float32)
    
    rotate_volume(volume_contiguous, rotated_contiguous, quat)
    rotated_int = rotated_contiguous.astype(np.int32)
    
    # Project
    proj = projection(rotated_int, lookup, voxel_size, config)
    
    # Stream to PV if requested
    if stream_pv:
        _STREAM_MANAGER.update_pv(stream_pv, proj)
    
    return proj


def simulate_laminography_projection(volume_labels, lookup, voxel_size, rotation_deg, 
                                     tilt_deg, config, stream_pv=None):
    """
    Simulate laminography projection - optionally stream to PV.
    
    Parameters
    ----------
    volume_labels : np.ndarray
        Volume to project
    lookup : dict
        Material lookup table
    voxel_size : float
        Voxel size
    rotation_deg : float
        Rotation angle in degrees
    tilt_deg : float
        Tilt angle in degrees
    config : object
        Simulation configuration
    stream_pv : str, optional
        PV name to stream to (e.g., 'SIMLP:IMG'). If None, no streaming.
    
    Returns
    -------
    np.ndarray
        Projection image
    """
    # Build quaternion
    quat = build_quaternion(np.deg2rad(tilt_deg), np.deg2rad(rotation_deg))
    
    # Rotate volume
    rotated = np.empty_like(volume_labels, dtype=volume_labels.dtype)
    volume_contiguous = np.ascontiguousarray(volume_labels, dtype=np.float32)
    rotated_contiguous = np.ascontiguousarray(rotated, dtype=np.float32)
    
    rotate_volume(volume_contiguous, rotated_contiguous, quat)
    rotated_int = rotated_contiguous.astype(np.int32)
    
    # Project
    proj = projection(rotated_int, lookup, voxel_size, config)
    
    # Stream to PV if requested
    if stream_pv:
        _STREAM_MANAGER.update_pv(stream_pv, proj)
    
    return proj


def simulate_projection_series(volume_labels, lookup, voxel_size, angles_deg, 
                               tilt_deg, config, stream_pv='SIMPS:IMG'):
    """
    Simulate series of projections - streams each projection as it's computed.
    
    Parameters
    ----------
    volume_labels : np.ndarray
        Volume to project
    lookup : dict
        Material lookup table
    voxel_size : float
        Voxel size
    angles_deg : array-like
        Rotation angles in degrees
    tilt_deg : float
        Tilt angle in degrees (0 for tomography)
    config : object
        Simulation configuration
    stream_pv : str or None
        PV name to stream to. If None, no streaming.
    
    Returns
    -------
    np.ndarray
        Array of projections, shape (n_angles, height, width)
    """
    print(f"Starting projection series: {len(angles_deg)} angles, tilt={tilt_deg}°")
    print(f"Volume size: {volume_labels.shape}, Memory: {volume_labels.nbytes / 1e9:.2f} GB")
    
    if stream_pv:
        print(f"Streaming to PV: {stream_pv}")
        print(f"Open your viewer and connect to: {stream_pv}")
        # Setup PV with dummy image
        _STREAM_MANAGER.setup_pv(stream_pv)
        time.sleep(1)  # Give viewer time to connect
    
    # For many projections, use GPU manager to avoid repeated transfers
    if len(angles_deg) > 5:
        gpu_manager = GPUVolumeManager(volume_labels, lookup, voxel_size, config)
        
        projections = []
        try:
            for i, angle in enumerate(angles_deg):
                print(f"Processing angle {i+1}/{len(angles_deg)}: {angle:.1f}°")
                
                proj = gpu_manager.rotate_and_project(angle, tilt_deg)
                projections.append(proj)
                
                # Stream to PV
                if stream_pv:
                    _STREAM_MANAGER.update_pv(stream_pv, proj)
                
                # Free GPU memory periodically
                if i % 10 == 0:
                    cp.get_default_memory_pool().free_all_blocks()
        
        finally:
            gpu_manager.cleanup()
    
    else:
        # For few projections, use simple approach
        projections = []
        for i, angle in enumerate(angles_deg):
            print(f"Processing angle {i+1}/{len(angles_deg)}: {angle:.1f}°")
            
            if tilt_deg == 0:
                proj = simulate_tomography_projection(
                    volume_labels, lookup, voxel_size, angle, config, 
                    stream_pv=stream_pv
                )
            else:
                proj = simulate_laminography_projection(
                    volume_labels, lookup, voxel_size, angle, tilt_deg, config,
                    stream_pv=stream_pv
                )
            
            projections.append(proj)
    
    if stream_pv:
        print(f"\n✓ All projections streamed to {stream_pv}")
    
    return np.array(projections)


def check_gpu_memory():
    """Check available GPU memory before starting."""
    mempool = cp.get_default_memory_pool()
    total_bytes = mempool.total_bytes()
    used_bytes = mempool.used_bytes()
    free_bytes = total_bytes - used_bytes
    
    print(f"GPU Memory: {used_bytes/1e9:.2f} GB used, {free_bytes/1e9:.2f} GB free")
    return free_bytes


def estimate_memory_needed(volume_shape, n_projections):
    """Estimate GPU memory needed for simulation."""
    volume_bytes = np.prod(volume_shape) * 4  # float32
    # Need: original volume + rotated volume + material maps + projection buffers
    estimated_bytes = volume_bytes * 6  # Conservative estimate
    
    print(f"Estimated GPU memory needed: {estimated_bytes/1e9:.2f} GB")
    return estimated_bytes


def list_active_pvs():
    """List all active streaming PVs."""
    if not _STREAM_MANAGER.servers:
        print("No active PVs")
        return []
    
    print(f"Active streaming PVs ({len(_STREAM_MANAGER.servers)}):")
    for pv_name, info in _STREAM_MANAGER.servers.items():
        count = _STREAM_MANAGER.counters[pv_name]
        print(f"  {pv_name}: {count} frames streamed")
    
    return list(_STREAM_MANAGER.servers.keys())


def cleanup_streaming():
    """Cleanup all streaming servers."""
    _STREAM_MANAGER.cleanup()
    print("All streaming servers cleaned up")


# Convenience function for keeping server alive
def keep_streaming_alive():
    """
    Keep the PV server alive so viewers can continue to see the data.
    Call this at the end of your script.
    Press Ctrl+C to stop.
    """
    print("\n" + "="*70)
    print("PV Server is running")
    print("="*70)
    list_active_pvs()
    print("\nYour viewer should now be able to see the PVs.")
    print("Press Ctrl+C to stop the server.")
    print("="*70)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
        cleanup_streaming()


if __name__ == '__main__':
    """Test streaming with random data."""
    print("Testing PV streaming...")
    
    # Create test volume
    test_volume = np.random.randint(0, 10, (50, 50, 50), dtype=np.int32)
    test_lookup = {i: {'density': i*0.1} for i in range(10)}
    test_config = type('Config', (), {'energy_kev': 5.0})()
    
    # Test single projection with streaming
    print("\nTest 1: Single projection")
    proj = simulate_tomography_projection(
        test_volume, test_lookup, 1.0, 45.0, test_config,
        stream_pv='TEST:SINGLE'
    )
    print(f"Projection shape: {proj.shape}")
    
    # Test series with streaming
    print("\nTest 2: Projection series")
    angles = np.linspace(0, 180, 10)
    projs = simulate_projection_series(
        test_volume, test_lookup, 1.0, angles, 0, test_config,
        stream_pv='TEST:SERIES'
    )
    print(f"Series shape: {projs.shape}")
    
    # Keep alive
    keep_streaming_alive()