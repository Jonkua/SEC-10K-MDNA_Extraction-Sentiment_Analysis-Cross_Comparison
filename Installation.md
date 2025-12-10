# Installation Guide
# Complete MD&A Sentiment Analysis Pipeline

## Quick Start

```bash
# 1. Clone or download the repository
git clone [your-repo-url]
cd sentiment-analysis-pipeline

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import torch, transformers, pandas; print('✓ All packages installed')"
```

## Individual Tool Requirements

### 1. MD&A Extractor Only
```bash
pip install pandas numpy beautifulsoup4 lxml colorlog nltk
```

### 2. Text Deduplicator Only
```bash
# No additional packages required - uses Python standard library only
```

### 3. FinBERT Analyzer Only
```bash
pip install torch transformers nltk pandas numpy

# For GPU support (NVIDIA CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. LM Dictionary Analyzer Only
```bash
pip install nltk textblob pandas numpy
```

### 5. Cross-Comparison Tool Only
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## GPU Support (Optional but Recommended for FinBERT)

### Check GPU Availability
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only (no GPU):**
```bash
pip install torch
```

**Check PyTorch Installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## External Data Files

### Loughran-McDonald Dictionary

**Required for:** LM Dictionary Analyzer

**Download:**
1. Visit: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
2. Download: `LoughranMcDonald_MasterDictionary_1993-2024.csv`
3. Place in accessible directory
4. Specify path when running LM analyzer

**Alternative sources:**
- Direct download: https://sraf.nd.edu/data/
- Academic mirror: Contact University of Notre Dame SRAF

## NLTK Data Download

NLTK data is downloaded automatically on first run. If manual download is needed:

```bash
python -m nltk.downloader punkt punkt_tab stopwords
```

Or download via Python:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## Verification Tests

### Test MD&A Extractor
```bash
python -c "from pathlib import Path; import pandas as pd; print('✓ MD&A Extractor dependencies OK')"
```

### Test FinBERT Analyzer
```bash
python -c "import torch; from transformers import BertTokenizer; print('✓ FinBERT Analyzer dependencies OK')"
```

### Test LM Analyzer
```bash
python -c "import nltk; from textblob import TextBlob; print('✓ LM Analyzer dependencies OK')"
```

### Test Cross-Comparison
```bash
python -c "import matplotlib.pyplot as plt; from scipy.stats import pearsonr; print('✓ Cross-Comparison dependencies OK')"
```

## System Requirements

### Minimum Requirements
- Python: 3.7+
- CPU: 4 cores
- RAM: 8GB
- Disk: 5GB
- OS: Windows 10, macOS 10.14+, or Linux

### Recommended Requirements
- Python: 3.9+
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM
- Disk: 10GB SSD
- OS: Windows 11, macOS 12+, or Ubuntu 20.04+

## Troubleshooting

### Issue: pip install fails with "externally-managed-environment"

**On Linux/Mac (Python 3.11+):**
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use --break-system-packages (not recommended)
pip install -r requirements.txt --break-system-packages
```

### Issue: torch installation is very slow

**Solution: Use --no-cache-dir**
```bash
pip install --no-cache-dir torch transformers
```

### Issue: CUDA version mismatch

**Check CUDA version:**
```bash
nvidia-smi
```

**Install matching PyTorch:**
- CUDA 11.x: Use cu118 wheel
- CUDA 12.x: Use cu121 wheel
- No GPU: Use CPU-only wheel

**Visit:** https://pytorch.org/get-started/locally/

### Issue: Memory error during FinBERT analysis

**Solutions:**
1. Reduce batch size in code
2. Use CPU-only mode (slower but less memory)
3. Process files in smaller batches
4. Increase system RAM or swap space

### Issue: NLTK punkt not found

**Solution:**
```bash
python -m nltk.downloader punkt punkt_tab stopwords
```

### Issue: ImportError: cannot import name 'TextBlob'

**Solution:**
```bash
pip install --upgrade textblob
python -m textblob.download_corpora
```

### Issue: matplotlib cannot display plots

**On Linux without GUI:**
```bash
# Add to script:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## Development Setup

For development and testing:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/

# Check code style
black --check .
flake8 .

# Type checking
mypy --strict .
```

## Docker Setup (Alternative)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab stopwords

# Copy application
COPY . .

CMD ["/bin/bash"]
```

**Build and run:**
```bash
docker build -t sentiment-analysis .
docker run -it -v $(pwd)/data:/app/data sentiment-analysis
```

## Conda Setup (Alternative)

```bash
# Create environment
conda create -n sentiment python=3.9

# Activate environment
conda activate sentiment

# Install packages via conda
conda install pandas numpy matplotlib seaborn scipy scikit-learn

# Install remaining via pip
pip install torch transformers nltk textblob beautifulsoup4 lxml colorlog
```

## Platform-Specific Notes

### Windows
- Use `venv\Scripts\activate` (not `source`)
- Long path support may need to be enabled
- Some packages may require Visual C++ Build Tools

### macOS
- Use homebrew Python: `brew install python@3.9`
- May need Xcode Command Line Tools: `xcode-select --install`

### Linux
- Use system Python or pyenv
- May need python3-dev package: `sudo apt-get install python3-dev`

## Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade pandas

# Check for outdated packages
pip list --outdated
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Or remove all packages (if not using venv)
pip uninstall -r requirements.txt -y
```

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review individual tool README files
3. Search for error messages online
4. Check package documentation:
   - PyTorch: https://pytorch.org/docs/
   - Transformers: https://huggingface.co/docs/transformers/
   - NLTK: https://www.nltk.org/
5. Open an issue on the repository

## Version Compatibility Matrix

| Python | PyTorch | Transformers | Pandas | Status |
|--------|---------|--------------|--------|--------|
| 3.7 | 1.9.0 | 4.0.0 | 1.3.0 | Minimum |
| 3.8 | 1.11.0 | 4.15.0 | 1.4.0 | Tested |
| 3.9 | 1.13.0 | 4.25.0 | 1.5.0 | Recommended |
| 3.10 | 2.0.0 | 4.30.0 | 2.0.0 | Tested |
| 3.11 | 2.1.0 | 4.35.0 | 2.1.0 | Latest |

## Success Checklist

- [ ] Python 3.7+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed successfully
- [ ] PyTorch recognizes GPU (if applicable)
- [ ] NLTK data downloaded
- [ ] LM Dictionary CSV downloaded (for LM Analyzer)
- [ ] All verification tests pass
- [ ] Sample script runs without errors

## Next Steps

After successful installation:

1. Review individual tool README files
2. Prepare your input data
3. Download LM Dictionary (for LM analysis)
4. Test with sample data
5. Run full analysis pipeline

**Installation complete! You're ready to analyze MD&A sentiment.**