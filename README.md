# vietnamese_document_classification
Vietnamese document classification

# Topics:
- Chinh tri xa hoi
- Doi song
- Khoa hoc
- Kinh doanh
- Phap luat
- Suc khoe
- The gioi
- The thao
- Van hoa
- Vi tinh

# Install
- Install via cmd 'pip install -r requirements.txt'
# Run options
- GUI (encouragement): main.py server
- Predict single file text: main.py predict file_path
- Continue train: main.y train
# Data
- **Optional** raw data [3].
- Download PhoBERT [2] to data/PhoBERT_base_fairseq
- Download training data and labels from [1] (https://drive.google.com/drive/folders/1stRredI0fZ2vE5_SKGggrgDxnV1bxhr1) to data/
- Download pretrain model from https://drive.google.com/drive/folders/1-gRQ3w01BJSZshuxh-xL1aBXycY-F1ui to data/checkpoint/latest_checkpoint.pt

# Reference
- [1] https://phamdinhkhanh.github.io/2020/06/04/PhoBERT_Fairseq.html
- [2] https://github.com/VinAIResearch/PhoBERT#fairseq
- [3] https://github.com/duyvuleo/VNTC
