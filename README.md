# RKNN implementation of tasks

Brief description of your project.

## About

Provide a more detailed description of your project. Explain what problem it solves, its purpose, and any background information that might be helpful for users.


### Steps

```bash
# Clone the repository
git clone -b rknn https://github.com/bobaimo/yolo_detection_docker.git

# Install dependencies
pip install ultralytics rknn-toolkit-lite2

# or for Python projects
curl -L -o librknnrt.so "https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

# replace the librknnrt.so file
sudo mv librknnrt.so /usr/lib/
