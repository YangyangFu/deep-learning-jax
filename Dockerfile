FROM tensorflow/tensorflow

USER root
RUN pip install pandas scikit-learn matplotlib emoji ipykernel tqdm faker pydub transformers[tf-cpu]