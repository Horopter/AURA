# FVC Dataset - 600 Videos

This dataset contains 300 fake AI-generated videos and 300 real videos for binary classification.

## Dataset Structure

```
videos/
├── fake/          # 300 fake AI-generated videos
└── real/          # 300 real videos

data/
└── dataset_600_videos.csv  # Metadata CSV with video paths and labels
```

## Dataset Statistics

- **Total Videos**: 600
- **Fake Videos**: 300
- **Real Videos**: 300
- **Total Size**: ~25 GB
- **Format**: MP4

## CSV Format

The `dataset_600_videos.csv` file contains:
- `video_id`: YouTube video ID
- `video_path`: Relative path to video file (e.g., `videos/fake/video_id.mp4`)
- `video_url`: Original YouTube URL
- `label`: "fake" or "real"

## Usage

The videos are organized in the `videos/` directory:
- `videos/fake/` - Contains all fake AI-generated videos
- `videos/real/` - Contains all real videos

## Note

Due to the large size (~25 GB), videos are not stored in this repository. 
Please download the videos separately or use the provided download script.

## Download Script

To download the videos, use:
```bash
python3 src/setup_fvc_dataset.py
```

Or use the scraper script from the techjam directory:
```bash
python3 simple_dataset_scraper.py
```

