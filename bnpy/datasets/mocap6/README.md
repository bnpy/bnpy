This folder holds a small annotated dataset of motion capture sensor traces from subjects performing simple exercises (jogging, arm circles, toe touches, etc.).

## Source

This data is an exact copy of the 6-sequence dataset released by Mike Hughes at this URL:

<https://github.com/michaelchughes/mocap6dataset>

Academic Paper:

```
"Joint Modeling of Multiple Time Series via the Beta Process with Application to Motion Capture Segmentation".
Fox, Hughes, Sudderth, and Jordan
AOAS 2014
```

* Publisher website: https://projecteuclid.org/euclid.aoas/1414091214
* PDF: https://www.michaelchughes.com/papers/FoxHughesSudderthJordan_AOAS_2014.pdf

Citation:

```
@article{foxJointModelingMultiple2014a,
  author = {Fox, Emily B. and Hughes, Michael C. and Sudderth, Erik B. and Jordan, Michael I.},
  title = {Joint Modeling of Multiple Time Series via the Beta Process with Application to Motion Capture Segmentation},
  journal = {Annals of Applied Statistics},
  year = {2014},
  volume = {8},
  number = {3},
  pages = {1281-1313}
}
```

## Dataset Summary

Six sequences were collected from files available at mocap.cs.cmu.edu:

Subject 13: trials 29, 30, and 31
Subject 14: trials 6, 14, and 20

Each of the six sequences has been annotated to indicate which of a set of 12 possible exercises is being performed at each timestep.

The raw AMC mocap sensor data from these sequences was post-processed as follows:

* 12 sensor channels were kept as representative of gross motor behavior. Remaining channels were discarded.
* Each sensor channel was adjusted to have zero-mean.
* Each channel was block-averaged to a final frame rate of 10 fps (down from 120 fps in the raw data).
