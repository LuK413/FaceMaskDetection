# FaceMaskDetection
To run this, begin by cloning the repository. Then, install the requirements using
```
pip install -r requirements.txt
```
After that, the data can be installed from Kaggle. If you have an account and an API key, use the following command
```
kaggle datasets download -d andrewmvd/face-mask-detection 
```
Otherwise, the data can be accessed via the following [link](https://www.kaggle.com/andrewmvd/face-mask-detection).
The data should then be unzipped and then renamed to 'facemask' and moved to the datasets folder.
Run the Jupyter Notebook to see the results.

### Dataset Credits
@misc{make ml,
title={Mask Dataset},
url={https://makeml.app/datasets/mask},
journal={Make ML}
}