# Controls

Sashimi uses a lot of keys to change parameters or perform actions. Here is the complete list and what they do :

## Camera Movements:

| key (QWERTY)  | key (AZERTY) | works when scanning | effect                                                   |
|:-------------:|:------------:|---------------------|----------------------------------------------------------|
|      `Q`      |     `A`      |                     | Move the camera upwards 1mm (+`Shift` is 10mm)           |
|      `E`      |     `E`      |                     | Move the camera downwards 1mm (+`Shift` is 10mm)         |
|      `W`      |     `Z`      |                     | Move the camera towards the back 1mm (+`Shift` is 10mm)  |
|      `A`      |     `Q`      |                     | Move the camera towards the left 1mm (+`Shift` is 10mm)  |
|      `S`      |     `S`      |                     | Move the camera towards the back 1mm (+`Shift` is 10mm)  |
|      `D`      |     `D`      |                     | Move the camera towards the right 1mm (+`Shift` is 10mm) |
| `Shift` + `C` |      --      |                     | Make the scanner find focus automatically                |
| `Shift` + `J` |      --      |                     | Go to the front-left corner of the current region        |
| `Shift` + `I` |      --      |                     | Go to the back-right corner of the current region        |
| `Shift` + `U` |      --      |                     | Go to the back-left corner of the current region         |
| `Shift` + `K` |      --      |                     | Go to the front-right corner of the current region       |
| `Shift` + `H` |      --      |                     | Go to the Home point                                     |

## Scan settings:

| key (QWERTY)  | key (AZERTY) | works when scanning | effect                                                                   |
|:-------------:|:------------:|:-------------------:|--------------------------------------------------------------------------|
|      `P`      |      --      |         yes         | Start or stop the scan sequence                                          |
|     `Esc`     |      --      |         yes         | Quit Sashimi                                                             |
|      `J`      |      --      |                     | Set the front-left corner of the region to the camera's current position |
|      `I`      |      --      |                     | Set the back-right corner of the region to the camera's current position |
|      `U`      |      --      |                     | Set the height of the back-left corner to the camera's current height    |
|      `Z`      |     `W`      |                     | Select the previous scan region                                          |
|      `X`      |      --      |                     | Select the next scan region                                              |
|      `V`      |      --      |                     | create a new scan region at the end of the sequence                      |
| `Shift` + `B` |      --      |                     | Delete the currently selected scan region                                |
| `Shift` + `N` |      --      |                     | Delete all of the scan regions                                           |
|      `G`      |      --      |         yes         | decrease the exposure by 50 µs                                           |
|      `T`      |      --      |         yes         | increase exposure by 50 µs                                               |
|      `[`      |      --      |         yes         | decrease the height of the stack by 100 µm                               |
|      `]`      |      --      |         yes         | increase the height of the stack by 100 µm                               |
|      `{`      |      --      |         yes         | decrease the vertical distance between two images by 20 µm               |
|      `}`      |      --      |         yes         | increase the vertical distance between two images by 20 µm               |
|      `H`      |      --      |                     | set the position of the home Point                                       |

## Other:

|    key (QWERTY)     | key (AZERTY)  | works when scanning | effect                                              |
|:-------------------:|:-------------:|:-------------------:|-----------------------------------------------------|
|     `?` or `/`      |      --       |         yes         | Show the help menu                                  |
| `Enter` or `Return` |      --       |                     | Make a single stack of images at the current point  |
|         `1`         | `Shift` + `1` |         yes         | Display camera feed with all three (RGB) components |
|         `2`         | `Shift` + `2` |         yes         | Display camera feed with only the red component     |
|         `3`         | `Shift` + `3` |         yes         | Display camera feed with only the green component   |
|         `4`         | `Shift` + `4` |         yes         | Display camera feed with only the blue component    |
