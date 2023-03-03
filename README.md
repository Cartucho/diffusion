# diffusion

How to run the code:

```
python3.8 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

You also need a folder containing a dataset of images, in my case I did with >6k images of cats, from [this kaggle dataset](https://www.kaggle.com/datasets/crawford/cat-dataset).
Only images are necessary. I downloaded that dataset and moved all the `.jpg` files into a single folder called `cats/`.

You can also play around with the settings of the diffusion by editing the `config.py` file.
