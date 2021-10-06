# Oemer (End-to-end OMR)

End-to-end Optical Music Recognition system build on deep learning models and machine learning techniques.
Default to use **Onnxruntime** for model inference. If you want to use **tensorflow** for the inference,
run `export INFERENCE_WITH_TF=true` and make sure there is TF installed.

![](figures/tabi_mix.jpg)

https://user-images.githubusercontent.com/24308057/136168551-2e705c2d-8cf5-4063-826f-0e179f54c772.mp4



## Quick Start
``` bash
git clone https://github.com/meteo-team/oemer
cd oemer
python setup.py install
oemer --help
```

Or download the built wheel file from the release and install
``` bash
# Go to the release page and download the .whl file from
# the assets.

# Replace the <version> to the correct version.
pip install Oemer-<version>-py3-none-any.whl
```

## Packaging
``` bash
python setup.py bdist_wheel

# Install from the wheel file
pip install dist/Oemer-<version>-py3-none-any.whl
```

## Change log level
``` bash
# Available options: debug, info, warn, warning, error, crtical
export LOG_LEVEL=debug
```


## Technical Details

This section describes the detail techniques for solving the OMR problem. The overall flow can also be found in [oemer/ete.py](https://github.com/meteo-team/oemer/blob/main/oemer/ete.py), which is also the entrypoint for `oemer` command.

### Model Prediction
Oemer first predicts different informations with two image semantic segmentation models: one for
predicting stafflines and all other symbols; and second model for more detailed symbol informations,
including noteheads, clefs, stems, rests, sharp, flat, natural.


<p align='center'>
    <img width="70%" src="figures/tabi_model1.jpg">
    <p align='center'>Model one for predicting stafflines (red) and all other symbols (blue).</p>
</p>
<p align='center'>
    <img width="70%" src="figures/tabi_model2.jpg">
    <p align='center'>Model two for predicting noteheads (green), clefs/sharp/flat/natural (pink), and stems/rests (blue).</p>
</p>

### Dewarping

Before proceed to recognizing the symbols, one may need to deskew the photo first since 
the later process assumes the stafflines are all horizontally aligned and the position 
of noteheads, rests and all other things are all depends on this assumption.

For the dewarping, there can be summarized to six steps as shown in the below figure.

<p align='center'>
    <img width="100%" src="figures/dewarp_steps.png">
    <p align='center'>Steps to dewarp the skewed image.</p>
</p>
