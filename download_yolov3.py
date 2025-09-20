import urllib3
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_yolov3_weights():
    # List of mirror URLs to try, in order
    urls = [
        "https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights",
        "https://drive.google.com/uc?export=download&id=1QsVOxH8_YstLwEFp2iF6lqpE2YE9U5Y6",
        "https://pjreddie.com/media/files/yolov3.weights"
    ]
    
    dest = "yolov3.weights"
    min_size = 200 * 1024 * 1024  # 200MB minimum
    
    # Remove existing small/invalid file
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size < min_size:
            logger.warning(f"Removing invalid weights file ({size} bytes)")
            os.remove(dest)
    
    # Configure urllib3 with retries and longer timeout
    http = urllib3.PoolManager(
        retries=urllib3.Retry(3, backoff_factor=0.5),
        timeout=urllib3.Timeout(connect=5, read=120)
    )
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }
    
    for url in urls:
        logger.info(f"Trying download from: {url}")
        try:
            with http.request('GET', url, headers=headers, preload_content=False) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to download from {url} (status {resp.status})")
                    continue
                
                file_size = int(resp.headers.get('Content-Length', 0))
                if file_size < min_size:
                    logger.warning(f"File too small ({file_size} bytes), trying next mirror")
                    continue
                
                logger.info(f"Downloading {file_size/1024/1024:.1f} MB...")
                
                with open(dest, 'wb') as f:
                    downloaded = 0
                    for chunk in resp.stream(32*1024):
                        downloaded += len(chunk)
                        f.write(chunk)
                        done = int(50 * downloaded / file_size)
                        sys.stdout.write(f'\rProgress: [{"="*done}{" "*(50-done)}] {downloaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB')
                        sys.stdout.flush()
                print()  # New line after progress bar
                
                if os.path.getsize(dest) >= min_size:
                    logger.info(f"Successfully downloaded {dest}")
                    return True
                else:
                    logger.warning("Downloaded file is too small, will try next mirror")
                    os.remove(dest)
            
        except Exception as e:
            logger.warning(f"Error downloading from {url}: {e}")
            continue
    
    logger.error("Failed to download weights from all mirrors")
    logger.error("\nPlease try downloading manually:")
    logger.error("1. Open this link in your browser:")
    logger.error("   https://drive.google.com/file/d/1QsVOxH8_YstLwEFp2iF6lqpE2YE9U5Y6/view?usp=sharing")
    logger.error("2. Click the Download button")
    logger.error("3. Move the downloaded file to:")
    logger.error(f"   {os.path.abspath(dest)}")
    return False

if __name__ == '__main__':
    download_yolov3_weights()