"""
Comprehensive unit tests for data/scan module.
Tests video scanning functions.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from lib.data.scan import (
    find_video_file,
    _probe_video_stats_ffprobe,
    _probe_video_stats_opencv,
    _probe_video_stats,
    scan_videos,
)


class TestFindVideoFile:
    """Tests for find_video_file function."""
    
    def test_find_video_file_exists(self, temp_dir):
        """Test find_video_file finds video file."""
        video_dir = Path(temp_dir) / "video_folder"
        video_dir.mkdir()
        video_file = video_dir / "video.mp4"
        video_file.write_text("dummy")
        
        result = find_video_file(str(video_dir))
        
        assert result is not None
        assert "video.mp4" in result
    
    def test_find_video_file_not_exists(self, temp_dir):
        """Test find_video_file returns None when no video found."""
        empty_dir = Path(temp_dir) / "empty_folder"
        empty_dir.mkdir()
        
        result = find_video_file(str(empty_dir))
        
        assert result is None


class TestProbeVideoStats:
    """Tests for video stats probing functions."""
    
    @patch('lib.data.scan.subprocess.run')
    def test_probe_video_stats_ffprobe_success(self, mock_subprocess, temp_dir):
        """Test _probe_video_stats_ffprobe with successful ffprobe."""
        import json
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {
                "duration": "10.5",
                "bit_rate": "1000000"
            },
            "streams": [{
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1"
            }]
        }).encode('utf-8')
        mock_result.stderr = b""
        mock_subprocess.return_value = mock_result
        
        test_video = Path(temp_dir) / "test.mp4"
        test_video.write_text("dummy")
        
        stats = _probe_video_stats_ffprobe(str(test_video))
        
        assert stats["width"] == 1920
        assert stats["height"] == 1080
        assert stats["fps"] == 30.0
    
    @patch('lib.data.scan.cv2.VideoCapture')
    def test_probe_video_stats_opencv(self, mock_cv2, temp_dir):
        """Test _probe_video_stats_opencv with mocked OpenCV."""
        mock_cap = Mock()
        mock_cap.get.return_value = 1920.0
        mock_cap.set = Mock()
        mock_cap.read.return_value = (True, None)
        mock_cv2.return_value = mock_cap
        
        test_video = Path(temp_dir) / "test.mp4"
        test_video.write_text("dummy")
        
        stats = _probe_video_stats_opencv(str(test_video))
        
        assert isinstance(stats, dict)
        assert "width" in stats


class TestScanVideos:
    """Tests for scan_videos function."""
    
    @patch('lib.data.scan.find_video_file')
    @patch('lib.data.scan._probe_video_stats')
    def test_scan_videos_basic(self, mock_probe, mock_find, temp_dir):
        """Test scan_videos returns list of video records."""
        from lib.data.config import FVCConfig
        
        mock_find.return_value = str(Path(temp_dir) / "video.mp4")
        mock_probe.return_value = {
            "width": 1920,
            "height": 1080,
            "fps": 30.0
        }
        
        cfg = FVCConfig(root_dir=temp_dir)
        # Create dummy video folder structure
        videos_dir = Path(cfg.videos_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)
        (videos_dir / "FVC1").mkdir()
        (videos_dir / "FVC1" / "video1").mkdir()
        
        records = scan_videos(cfg, compute_stats=False)
        
        assert isinstance(records, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
