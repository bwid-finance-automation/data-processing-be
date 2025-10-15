"""
File utility functions for billing module
"""
from pathlib import Path
from typing import List
import aiofiles
from fastapi import UploadFile, HTTPException
from .logging_config import get_logger

logger = get_logger(__name__)


async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save an uploaded file to destination"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(destination, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)

        logger.info(f"File saved: {destination.name} ({len(content)} bytes)")
        return destination

    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file extension is allowed"""
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def get_files_in_directory(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Get all files in directory with optional extension filter"""
    if not directory.exists():
        return []

    files = []
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith('~$'):
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(file_path)

    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes"""
    if file_path.exists():
        return file_path.stat().st_size
    return 0


def delete_file(file_path: Path) -> bool:
    """Delete a file"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"File deleted: {file_path.name}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return False
