# Image Classifier 3D

[![Build Status](https://github.com/aics-int/image_classifier_3d/workflows/Build%20Master/badge.svg)](https://github.com/aics-int/image_classifier_3d/actions)

Python package for building 3d image classifier using deep neural networks

This was used to build the cell classifier to automatically annotate the cells (as interphase, prophase, metaphase, anaphase/telophase or outliers) in hIPSC single cell dataset from Allen Institute for Cell Science.

---

## Installation

**Stable Release:** `pip install image_classifier_3d`<br>

**Install nightly development head and make customization:**

```bash
git clone git@github.com:AllenCell/image_classifier_3d.git
cd image_classifier_3d
pip install -e .[all]
```

## Documentation

### Quick Start:

1. Training 

Set up a yaml file to specify training configurations. See an example [HERE](config_examples/config_train.yaml) with embedded instructions.

```console
run_classifier3d --config /path/to/config_train.yaml --mode train --debug
```

2. Testing (with known labels, reporting performance)

Set up a yaml file to specify testing configurations. See an example [HERE](model_zoo/test_config_mitotic_classifier.yaml) with embedded instructions.

```console
run_classifier3d --config /path/to/config_test.yaml --mode validate --debug
```

3. Inference (without labels, annotating new cells)

It is possible to run inference as training/testing, just with `--mode inference`. However, it might be more common that you want to call the classifier in other python scripts, for example, as one step of you bigger workflow. 

```python
my_classifier = ProjectTester(save_model_output=False)
df_pred = my_classifier.run_tester_csv(dataset, pred_path, return_df=True)
```

Here, `dataset` is a filename pointing to a csv file listing all the images to be applied on. See [`CSV` mode in `dataloader`](TODO) for details. `pred_path` is a filepath to save intermediate prediction tables (e.g., predictions from each individual model when using ensemble). The final labels are returned to a dataframe `df_pred`.


### Full Documentation:

For full package documentation please visit [AllenCell.github.io/image_classifier_3d](https://AllenCell.github.io/image_classifier_3d).


## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## Questions?

If you have any questions, feel free to leave a comment in our Allen Cell forum: [https://forum.allencell.org/](https://forum.allencell.org/). 

***Free software: Allen Institute Software License***
