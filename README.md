# flowvis
Visualize a 2D vector field stored in the `flow.raw` file.

## Keymappings

| Key                       | Action                                                           |
|---------------------------|------------------------------------------------------------------|
| <kbd>ctrl +</kbd>         | Zoom in                                                          |
| <kbd>ctrl -</kbd>         | Zoom out                                                         |
| <kbd>ctrl backspace</kbd> | Reset pan and zoom                                               |
|                           |                                                                  |
| <kbd>Space</kbd>          | Play/pause video playback                                        |
| <kbd>.</kbd>              | Step to next frame                                               |
| <kbd>,</kbd>              | Step to previous frame                                           |
|                           |                                                                  |
| <kbd>1</kbd>              | Show/hide the background velocity texture                        |
| <kbd>ctrl 1</kbd>         | Cycle through the color map for the background texture           |
| <kbd>F</kbd>              | Enable/disable linear filtering of the background texture        |
|                           |                                                                  |
| <kbd>2</kbd>              | Show/hide stream lines                                           |
| <kbd>ctrl 2</kbd>         | Cycle through the color map for stream lines                     |
| <kbd>I</kbd>              | Enable/disable interactive an stream line at the cursor position |
| <kbd>M</kbd>              | Cycle the stream line method between `Euler`, `RK2` and `RK4`    |
| <kbd>L</kbd>              | Spawn stream lines equally distributed on the left               |
| <kbd>Shift L</kbd>        | Spawn stream lines more concentrated towards the center left     |
| <kbd>Delete</kbd>         | Remove all stream lines                                          |
|                           |                                                                  |
| <kbd>3</kbd>              | Show/hide the arrow plot                                         |
| <kbd>ctrl 3</kbd>         | Cycle through the color map for the arrow plot                   |
| <kbd>[</kbd>              | Increase arrow density                                           |
| <kbd>]</kbd>              | Decrease arrow density                                           |

## Mouse control

| Key                              | Action                   |
|----------------------------------|--------------------------|
| <kbd>ctrl left-click</kbd>       | Add a stream line origin |
| <kbd>ctrl right-click</kbd>      | Remove a stream line     |
| <kbd>ctrl mouse-wheel</kbd>      | Zoom                     |
| <kbd>middle-click</kbd> and drag | Pan                      |

