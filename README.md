# hdr10_tool
Utility to analyze HDR10 (PQ) luminance and MaxCLL/MaxFALL from video files.

## Supported formats
MP4, MKV, MOV (via OpenCV decoding)

## Usage

### Analyze a video (global results only)
Prints a progress bar and global MaxCLL / MaxFALL values.

```bash
python hdr10_tool.py analyze -i input_video.mkv
````

### Generate HDR10 PQ plot

Analyzes the video and saves a luminance plot as PNG.

```bash
python hdr10_tool.py plot -i input_video.mkv -o output_plot.png
```
