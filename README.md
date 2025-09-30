# Neural Network Based Table Structure Recognition
Code for neural network based table structure recognition, based on Annika Thelen's Bachelor Thesis.

# Used Datasets
GloSAT:https://github.com/stuartemiddleton/glosat_table_dataset/tree/main

Wired Table in the Wild (WTW): https://tianchi.aliyun.com/dataset/108587

Bonner Tabellendatensatz BonnData

# Used Models
Kosmos 2.5: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

TableTransformer: https://huggingface.co/docs/transformers/model_doc/table-transformer

Faster R-CNN: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

# Dependencies
Please see https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/tree/main to preprocess GloSAT and BonnData Datasets



# Get started
The dataset currently can not be downloaded by a script. You need to clone https://gitlab.uni-bonn.de/digital-history/tabellenlayout/immediat-tables by hand. It needs to have the following structure, with _annotations_ containing all the Transkribus annotation -xml-files and _images_ containing all
the images as .jpg-files.

```
.
+-- data
|   +--BonnData
|   |  +--annoations
|   |  |  +--.xml-files
|   |  |        ...
|   |
|   |  +--images
|      |  +--.jpg-files
|      |        ...
```

### GloSAT dataset
The GloSAT dataset can be downloaded using the `download.py` script: 
```python
    python -m src.historicdocumentprocessing.download
```


## Preprocess the data
For preprocessing the data the `preprocess.py` can be used. It creates a new folder in data/BonnData
or data/GloSAT called preprocessed with all the preprocessed data. Use `--BonnData` or `--GloSAT` to
preprocess the specific dataset.

```python
    python -m src.historicdocumentprocessing.preprocess --BonnData
```
```python
    python -m src.historicdocumentprocessing.preprocess --GloSAT
```

## Create Training, Validation and Test split
To create a split on the data the `split.py` can be used:

```python
    python -m src.historicdocumentprocessing.split --BonnData
```
```python
    python -m src.historicdocumentprocessing.split --GloSAT
```

## Train a model
To train a model the `trainer.py` script can be used:
```python
    python -m src.historicdocumentprocessing.tabletransformer_train
```
Here the following parameter can be used:

| parameter           | functionality                                                  |
|---------------------|----------------------------------------------------------------|
| --name, -n          | name of the model in savefiles and logging                     |
| --epochs, -e        | Number of epochs to train                                      |
| --dataset, -d       | which dataset should be used for training (GloSAT or BonnData) |
| --objective, -o     | objective of the model ('table', 'cell', 'row' or 'col')       |
| --augmentations, -a | Use augmentations while training                               |


## Evaluate a model
To evaluate a model the `evaluation.py` script can be used:
```python
    python -m src.TableExtraction.evaluation
```

Here the following parameter can be used:

| parameter           | functionality                                                  |
|---------------------|----------------------------------------------------------------|
| --dataset, -d       | which dataset should be used for training (GloSAT or BonnData) |
| --objective, -o     | objective of the model ('table', 'cell', 'row' or 'col')       |
| --model, -m         | name of the model to load and evaluate                         |






# Used Code
Included Post-Processing from https://github.com/stuartemiddleton/glosat_table_dataset/blob/main/dla/src/table_structure_analysis.py
copyright:

Copyright (c) 2021, University of Southampton
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code or data must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. All advertising and publication materials containing results from the use
   of this software or data must acknowledge the University of Southampton
   and cite the following paper which describes the dataset:

   Ziomek. J. Middleton, S.E.
   GloSAT Historical Measurement Table Dataset: Enhanced Table Structure Recognition Annotation for Downstream Historical Data Rescue,
   6th International Workshop on Historical Document Imaging and Processing (HIP-2021),
   Sept 5-6, 2021, Lausanne, Switzerland

4. Neither the name of the University of Southampton nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE AND DATA IS PROVIDED BY University of Southampton ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL University of Southampton BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Modified Code from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
used https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR as a guideline for TableTransformer

License:
MIT License
Copyright (c) 2021 NielsRogge
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Modified and reused some code from my Projectgroup: https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/tree/main
