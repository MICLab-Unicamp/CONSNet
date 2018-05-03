FROM conda/miniconda2

RUN apt-get update && apt-get install -y git build-essential

RUN pip install --ignore-installed --upgrade scikit-learn https://github.com/mind/wheels/releases/download/tf1.6-cpu/tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl keras nibabel SimpleITK
RUN pip install --ignore-installed --upgrade git+git://github.com/rmsouza01/siamxt.git#egg=siamxt
RUN mkdir /code
COPY . /code/
