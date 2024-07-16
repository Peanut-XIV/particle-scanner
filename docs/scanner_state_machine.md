# The state machine in Sashimi's scanner

In order to keep the GUI responsive, Sashimi's scanner class is
now using a state machine. Each state contains work that can be done quickly, without
waiting. When the scanner needs to wait, the control in given back to the main
thread for the rest of the current frame.

Here is a graph representation of the state-machine :

```plantuml
state sharpness {
    state init_sharp_A {
    }
    state init_sharp_B {
    }
    state sharp_step {
    }
    state sharp_end {
    }

}


state main_loop {
    state Zone {
    }
    state stack_init {
    }
    state exposure {
    }
    state stack_z {
    }
    state image {
    }
}

hide empty description
state Idle #lightblue : wait for user action
Init: initialise internal variables
Zone: change the current scan zone\nGo to the first stack
Done: save relevent information\nwait for parallel jobs to finish
state "Stack Init" as stack_init:\
go to the current stack's position\ncalibrate the Z axis if necessary

state "Init Sharpness A" as init_sharp_A:\
measure the focus of\nthe scanners images
state "Init Sharpness B" as init_sharp_B:\
measure the gradient of\nfocus along the vertical axis
state "Sharpness Step" as sharp_step:\
measure the focus\nand compare to the\ngradient's direction
state "Sharpness End" as sharp_end:\
analyse the image\nat maximum focus:\ntry to detect forams

exposure:\
set the exposure to one\n of multiple possible values\n\
(allows to take a same stack\nat different exposures)

state "Stack Z" as stack_z: places the camera at the correct\nheight for the start of the stack
image: take a picture at a certain height,\nthen move a step up

Idle --> Init: user\npresses\nstart
Init --> Idle: zone count = 0
Init --> Zone: zone count > 0
Zone --> Done: remaining\nzones = 0
Done -u-> Idle
Zone -[#green]-> stack_init: remaining\nzones > 0

stack_init -[#green]-> exposure: DWS\nDisabled
stack_init -[#green]-> init_sharp_A: DWS\nEnabled
init_sharp_A -[#green]-> init_sharp_A: focus < threshold:\nmove upward fast
init_sharp_A -[#green]-> init_sharp_B: focus ≥ threshold:\nmove upward slowly
init_sharp_B -[#green]-> sharp_step: move slowly in\nthe gradients direction
sharp_step -[#green]-> sharp_step: maximum not found:\nkeep moving slowly
sharp_step -[#green]-> sharp_end: maximum found:\ngo back to the\nmaximum's position
sharp_end -[#green]-> exposure
init_sharp_A -[#red,dashed]-> sharp_end
init_sharp_B -[#red,dashed]-> sharp_end
sharp_step -[#red,dashed]-> sharp_end
sharp_end -[#red,dashed]-> exposure

exposure --> stack_init:\
remaining stacks > 0\n\
and remaining exp = 0\n\
or invalid stack

exposure -[#orange]-> stack_z: remaining\nexp > 0
stack_z -[#green]-> image
image -[#green]-> image: pictures\nremaining\n > 0
image --> exposure: pictures\nremaining\n = 0
exposure --> Zone: remaining\nexp = 0\nand\nremaining\nstacks = 0
```
```plantuml
state legend {
A -r-> B: change state\ndirectly
B -r[#green]-> C: wait for the end of\na movement before\nchanging state
A -d[#invisible]-> D
D -r[#orange]-> E: wait for the\nupdate of the\nexposure before\nchanging state
E -r[#red,dashed]-> F: change state\ndirectly because\nof an error
state "Start Node" as s_node #lightblue
}


```
## Glossary

DWS
: Detect While Scanning. Allows to search for the presence of objects of interest
within the stack, to then decide wether to take the stack or not.

Stack
: A set of pictures taken at a regular height intervals. A stack has a total
height (distance between the highest and lowest point) and a step (the distance
between two consecutive images).

