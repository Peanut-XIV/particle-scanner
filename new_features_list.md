# List of features added to Sashimi over the period of May and June 2023 :

### added the ability to designate a set of region to scan in sequence through the GUI

- create a new region with [V]
- Navigate between regions with [W] (previous) and [X] (next)
- Delete a region with Shift + [B] and delete all regions with Shift + [N]

---

### added the possibility to take every picture at several exposure durations

In the command prompt when starting sashimi, add `mult-exp` or `-e` to the command to enable
the feature. Then, follow by a list of values between quotation marks like so:

> sashimi scan --mult-exp "1000, 2000, 2100, 6000"

Not entering a list of values will make a prompt appear, asking for the minimum, 
maximum and interval between exposure values.

---

### added the option to focus stack the images automatically while scanning
This feature is enabled by default. If you wish to disable it when starting sashimi, 
add `--skip-fs` or `-s` to the command to skip the stacking step like so:

> sashimi scan --skip-fs

*Note: the stacking step is done in parallel to the scanning. 
Therefore, the scan will NOT be faster if the option is disabled*

---

### added a correction term to the vertical position of each stack

in order to account for the possible slope of the regions to scan, the height is measured
at three corners of each region. Then the correction has two settings :

- ***Lowest*** linearly interpolates the height of the fourth corner and chooses
the lowest height of the four. It becomes the starting height of every stack in the region
- ***Smart*** linearly interpolates the starting height of each stack according
to it's position in the region

"Smart" is the default option. If you wish to use "Lowest", you must add `--lowest` or `-z`
to the command when starting sashimi:

> sashimi scan --lowest

---

### in the GUI, changed the progress indicator to show the number of pictures taken over the total number of pictures in the current scan

---

### added the option to remove the raw, unstacked pictures after stacking them

This option is enabled by default. If you wish to disable it, add `--remove-raw` or `-r`
to the command when starting sashimi:

> sashimi scan --remove-raw

---

### added the option to quit the application automatically after scanning

This option is disabled by default. If you wish to enable it, add `--auto-quit` or `-q`
to the command:

> sashimi scan --auto-quit

---

### added a margin that is subtracted from the starting height of every stack

The height of the margin is 200 µm by default. If you wish to change its value, add `--margin` or `-m`, 
followed by the value you wish to replace it by in micrometers: 

> sashimi scan --margin 350

*Note: The value must be a relative integer. Positive means the stack starts lower.*

---

### the settings of the camera are now saved in a file located inside the sashimi package

It allows to keep the same color settings for every scan. Its name is nodeFile.pfs

---

### added an option to not overwrite existing folders

it is disabled by default. If you wish to enable it, add `--no` or `-n` to the command :

> sashimi scan --no

---

### separated language option from keyboard layout option

you can choose the language by adding `--lang` or `-l` followed by `fr` or `en`.
you can choose the layout by adding `--layout` followed by `AZERTY` or `QWERTY`.
`en` and `AZERTY` are the default values.

> sashimi scan --lang fr --layout QWERTY

---

### added a summary text file that is written at the end of a scan sequence.
It indicates the settings used, the duration of the scan, the number of pictures taken etc.
