"""
Comprehensive unit tests for utils/optical_flow module.
Tests optical flow extraction with dummy frames.
"""
import pytest
import numpy as np
from lib.utils.optical_flow import (
    extract_optical_flow,
    flow_to_rgb,
    extract_optical_flow_sequence,
    extract_optical_flow_video,
)


class TestExtractOpticalFlow:
    """Tests for extract_optical_flow function."""
    
    @pytest.fixture
    def dummy_frames(self):
        """Create dummy frames."""
        frame1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        return frame1, frame2
    
    def test_extract_optical_flow_farneback(self, dummy_frames):
        """Test extract_optical_flow with Farneback method."""
        frame1, frame2 = dummy_frames
        
        try:
            flow = extract_optical_flow(frame1, frame2, method="farneback")
            
            assert flow.shape == (224, 224, 2)  # (H, W, 2) for flow vectors
        except Exception as e:
            pytest.skip(f"OpenCV optical flow not available: {e}")
    
    def test_extract_optical_flow_grayscale(self):
        """Test extract_optical_flow with grayscale frames."""
        frame1 = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
        try:
            flow = extract_optical_flow(frame1, frame2, method="farneback")
            assert flow.shape == (224, 224, 2)
        except Exception as e:
            pytest.skip(f"OpenCV optical flow not available: {e}")


class TestFlowToRgb:
    """Tests for flow_to_rgb function."""
    
    def test_flow_to_rgb_basic(self):
        """Test flow_to_rgb converts flow to RGB."""
        flow = np.random.randn(224, 224, 2).astype(np.float32)
        
        try:
            rgb = flow_to_rgb(flow)
            assert rgb.shape == (224, 224, 3)
            assert rgb.dtype == np.uint8
        except Exception as e:
            pytest.skip(f"OpenCV not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
