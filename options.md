# Options

In order to use Sashimi, you need to run the following command in your python environment:

`python -m sashimi.cli scan --dir DIRECTORY/TO/SAVE/IMAGES --port PORT --lang en --layout QWERTY`

This command uses several options to modify the behavior of sashimi, which are preceded by `--` or `-` 
and followed by an argument if necessary. Although the options of this example are already explained in 
[READ ME](README.md), there are several others that you could find useful.

---

###  Multiple exposures: `--mult-exp`

You may need to take pictures at different exposures in order to compare them and find the best setting. 
In order to do that, add `--mult-exp` or `-e` to the command to enable
the feature. Then, follow by a list of values between quotation marks like so:

`python -m sashimi.cli scan [//OTHER OPTIONS//] --mult-exp "1000, 2000, 2100, 6000"`

You can also not give any argument. In this case, a prompt will appear, asking for the minimum, 
maximum and interval between exposure values.

---

### Skip the stacking step while scanning: `--skip-fs`
By default, Sashimi fuses the images automatically in parallel of scanning. You may want to disable this functionality
if the computer is too slow to scan and stack at the same time, if you wish to use an algorithm other than Helicon-Focus
or if you just do not need fused images at all. If you wish to disable it, add `--skip-fs` or `-s` to the command to 
skip the stacking step like so:

`python -m sashimi.cli scan [//OTHER OPTIONS//] --stack-fs`

*Note: the stacking step is done in parallel to the scanning.
Therefore, the scan will not be faster if the option is disabled*

---

### Change the correction term of the vertical position of each stack: `--lowest`

in order to account for the possible slope of the regions to scan, the height is measured at three corners of each 
region. Then the correction has two settings :

- **Lowest** linearly interpolates the height of the fourth corner and chooses the lowest height of the four. It becomes 
the starting height of every stack in the region. It is a good option if you are not sure of the background surface.
- **Linear** linearly interpolates the starting height of each stack according to it's position in the region. It works 
best if the background surface is flat but slanted.

**Linear** is the default option. If you wish to use **Lowest**, you must add `--lowest` or `-z` to the command:

`python -m sashimi.cli scan [//OTHER OPTIONS//] --lowest`

---

### Remove the raw, unstacked pictures after fusing them : `--remove-raw`

By default, Sashimi keeps the original pictures in order to preserve the raw data. If you wish to get rid of them 
automatically, add `--remove-raw` or `-r` to the command:

`python -m sashimi.cli scan [//OTHER OPTIONS//] --remove-raw`

---

### Quit the application automatically after scanning : `--auto-quit`

If you wish to run Sashimi over the night or the weekend and do not want it to keep running after it is finished, you 
can add `--auto-quit` or `-q`
to the command:

`python -m sashimi.cli scan [//OTHER OPTIONS//] --auto-quit`

---

### Change the vertical offset to the position of every stack: `--margin`

Since it is easier to set a position when the image is focused, a margin of 200 µm is subtracted to the vertical 
coordinate by default. If you wish to change its value, add `--margin` or `-m`, followed by the value you wish to 
replace it by in micrometers: 

`python -m sashimi.cli scan [//OTHER OPTIONS//] --margin 350`

*Note: The value must be a relative integer. Positive means the stack starts lower.*

---

### Keep Sashimi from overwriting existing data: `--no`

If you want Sashimi to stop if the given directory is not empty, add `--no` or `-n` to the command :

`python -m sashimi.cli scan [//OTHER OPTIONS//] --no`
