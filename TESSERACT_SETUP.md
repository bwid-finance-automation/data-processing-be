# Tesseract OCR Setup Guide

This guide will help you install and configure Tesseract OCR for the Contract OCR feature.

## Overview

The Contract OCR system now uses a two-step process:
1. **Tesseract OCR** extracts text from PDF/images locally
2. **OpenAI (GPT-4)** extracts structured data from the OCR text

This approach is more cost-effective than using OpenAI Vision API directly.

---

## Installation

### macOS

```bash
# Install Tesseract using Homebrew
brew install tesseract

# Verify installation
tesseract --version
```

### Ubuntu/Debian Linux

```bash
# Install Tesseract
sudo apt-get update
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

### Windows

1. Download the Tesseract installer from:
   https://github.com/UB-Mannheim/tesseract/wiki

2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.3.0.exe`)

3. During installation, note the installation path (e.g., `C:\Program Files\Tesseract-OCR`)

4. Add Tesseract to your `.env` file:
   ```
   TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
   ```

5. Verify installation:
   ```cmd
   tesseract --version
   ```

---

## Python Dependencies

Install the required Python packages:

```bash
cd data-processing-be
pip install -r requirements.txt
```

This will install:
- `pytesseract>=0.3.10` - Python wrapper for Tesseract
- `pillow>=10.0.0` - Image processing
- `pymupdf>=1.23.0` - PDF processing

---

## Configuration

### Environment Variables

Update your `.env` file based on `.env.example`:

```bash
# Required: OpenAI API key for text extraction
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o

# Optional: Tesseract path (only needed for Windows or custom installations)
# TESSERACT_CMD=/usr/local/bin/tesseract
```

### macOS/Linux
On macOS and Linux, Tesseract is usually in the system PATH after installation, so `TESSERACT_CMD` is not required.

### Windows
On Windows, you must set `TESSERACT_CMD` to point to the tesseract.exe location.

---

## Testing the Installation

### 1. Test Tesseract Installation

```bash
# Create a test image with text
echo "Hello World" | convert -pointsize 72 label:@- test.png

# Run OCR on the test image
tesseract test.png stdout
```

You should see "Hello World" in the output.

### 2. Test Python Integration

Create a test Python script:

```python
import pytesseract
from PIL import Image

# Test OCR
image = Image.new('RGB', (200, 100), color='white')
text = pytesseract.image_to_string(image)
print("Tesseract is working!")
```

### 3. Test Contract OCR Service

Start the backend server:

```bash
cd data-processing-be
uvicorn app.finance_accouting.main:app --reload --port 8000
```

Test the API:

```bash
curl http://localhost:8000/api/v1/contract-ocr/health
```

Upload a test contract via the frontend or API to verify the full pipeline.

---

## How It Works

### Previous Flow (Vision API)
```
PDF/Image → PyMUPDF → Base64 Image → OpenAI Vision → Extracted JSON
```

### New Flow (Tesseract + Text API)
```
PDF/Image → PyMUPDF → Tesseract OCR → Raw Text → OpenAI Text Model → Extracted JSON
```

### Processing Steps

1. **PDF Processing**:
   - PyMuPDF extracts text directly from text-based PDFs
   - If text is insufficient (< 50 chars), renders page as image and uses Tesseract
   - Processes up to 10 pages by default

2. **Image Processing**:
   - Uses Tesseract OCR with PSM mode 6 (uniform block of text)
   - Extracts all text from the image

3. **Text Extraction**:
   - Sends OCR text to OpenAI GPT-4 (text model)
   - Extracts 35+ structured fields using AI
   - Returns JSON with contract information

### Benefits

✅ **Cost Reduction**: Text API calls are ~10x cheaper than Vision API
✅ **Faster Processing**: Local OCR is faster than uploading images
✅ **Better Control**: Can preprocess/clean OCR text before AI
✅ **Transparency**: Raw OCR text is stored in `raw_text` field
✅ **Offline Capability**: OCR works without internet (only AI needs API)

---

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"

**macOS/Linux:**
```bash
which tesseract
# Should output: /usr/local/bin/tesseract or similar
```

If not found, install Tesseract using the instructions above.

**Windows:**
Add `TESSERACT_CMD` to your `.env` file with the full path to tesseract.exe.

### "No text could be extracted from the document"

- Check image quality (resolution should be at least 300 DPI)
- Ensure the image contains readable text
- Verify the file is not corrupted
- Try a different OCR PSM mode (edit line 52 in contract_ocr_service.py)

### Poor OCR Accuracy

- Increase image resolution (edit line 87: change `Matrix(2, 2)` to `Matrix(3, 3)`)
- Use higher quality source images
- Preprocess images (deskew, denoise, binarize)
- Try different Tesseract PSM modes:
  - `--psm 1`: Automatic page segmentation with OSD
  - `--psm 3`: Fully automatic page segmentation (default)
  - `--psm 6`: Assume uniform block of text (current)
  - `--psm 11`: Sparse text

### API Errors

Check the logs:
```bash
# Backend logs will show detailed OCR progress
# Look for lines like:
# "Extracted X characters from image"
# "Successfully extracted Y characters of text"
```

---

## Advanced Configuration

### Custom Tesseract Config

Modify the OCR config in [contract_ocr_service.py:52](app/finance_accouting/services/contract_ocr_service.py#L52):

```python
# Current config
text = pytesseract.image_to_string(image, config='--psm 6')

# For better accuracy with mixed layouts
text = pytesseract.image_to_string(image, config='--psm 3 --oem 1')

# For single-column documents
text = pytesseract.image_to_string(image, config='--psm 4')
```

### Adjust PDF Page Limit

Edit [contract_ocr_service.py:31](app/finance_accouting/services/contract_ocr_service.py#L31):

```python
self.pdf_max_pages = 20  # Process more pages
```

### Adjust Image Quality

Edit [contract_ocr_service.py:87](app/finance_accouting/services/contract_ocr_service.py#L87):

```python
# Higher quality (slower, larger)
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom

# Lower quality (faster, smaller)
pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom
```

---

## Language Support

Tesseract supports multiple languages. Install language packs:

### macOS
```bash
brew install tesseract-lang
```

### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr-eng tesseract-ocr-vie  # English + Vietnamese
```

### Using Multiple Languages

Update the OCR call in contract_ocr_service.py:

```python
# English + Vietnamese
text = pytesseract.image_to_string(image, lang='eng+vie', config='--psm 6')
```

---

## Performance Tips

1. **Use text-based PDFs when possible** - Direct text extraction is much faster than OCR
2. **Limit page count** - Only process necessary pages (default: 10)
3. **Optimize image quality** - Balance between quality and processing speed
4. **Batch processing** - Process multiple contracts in one API call
5. **Monitor logs** - Check processing times to identify bottlenecks

---

## Support

For issues or questions:
- Check the backend logs for detailed error messages
- Verify Tesseract installation: `tesseract --version`
- Ensure Python packages are installed: `pip list | grep tesseract`
- Test with a simple text image first before complex contracts

---

## References

- [Tesseract OCR Documentation](https://tesseract-ocr.github.io/)
- [PyTesseract GitHub](https://github.com/madmaze/pytesseract)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
