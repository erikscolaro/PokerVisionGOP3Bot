# GOP3ANTIBOT

## Description
GOP3ANTIBOT is an advanced image analysis tool designed for poker gameplay. It utilizes computer vision techniques to detect and interpret poker game elements from screen captures, including cards, player bets, and game stages. The system employs Tesseract OCR for text recognition and OpenCV for image processing and template matching.

## Features
- **Advanced Image Processing**: Uses OpenCV for preprocessing images with techniques like:
  - Grayscale conversion
  - Binary thresholding
  - Image scaling
  - Template matching for object detection (in the future will be replaced with ORB feature detection)
- **Powerful Text Recognition**: Implements Tesseract OCR with custom language models for poker-specific text
- **Game State Tracking**: Utilizes a finite state machine to track poker game progression through different stages (%TODO)
- **Template Matching**: Detects game elements by matching them against template images
- **Duplicate Detection Elimination**: Implements confidence and overlap thresholds to remove duplicate detections

## ⚠️ Warning: Work in Progress ⚠️
This project is currently under active development and not yet complete. Features may be missing, incomplete, or subject to significant changes. Use at your own risk and expect potential issues or limitations until a stable release is announced. Feel free to contribute with issues!